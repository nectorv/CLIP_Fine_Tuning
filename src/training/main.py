import torch
import bitsandbytes as bnb
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import wandb
import os

from src.training.arg_pars import parse_args
from src.training.utils import S3Manager, EarlyStopper, find_max_batch_size
from src.training.data import FurnitureDataset, get_transforms
from src.training.model import get_model
from src.config import TrainingRunConfig

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup WandB (Resume logic)
    wandb.init(project=args.wandb_project, config=args, resume="allow", name=f"{args.scenario}_lr{args.lr}")

    # 2. Data Prep
    s3_mgr = S3Manager(args.s3_bucket)
    s3_mgr.download_dataset(args.s3_prefix, args.data_dir)
    
    transform = get_transforms(args.augmentation)
    model, processor = get_model(args.scenario, args.lora_r)
    model.to(device)
    param_device = next(model.parameters()).device
    print(f"🧠 Model loaded on device: {param_device}")

    full_dataset = FurnitureDataset(args.data_dir, processor, transform=transform, mini=args.mini_train)

    num_samples = len(full_dataset)
    if num_samples == 0:
        raise ValueError(
            "No images found after S3 sync. Check --s3_bucket, --s3_prefix, and --data_dir."
        )
    if num_samples < 2:
        raise ValueError(
            "Dataset too small to split train/val. Need at least 2 images."
        )
    
    # Train/Val Split
    train_size = max(1, int(TrainingRunConfig.TRAIN_SPLIT * num_samples))
    val_size = num_samples - train_size
    if val_size == 0:
        val_size = 1
        train_size = num_samples - 1
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 3. Dynamic Batch Size
    if args.micro_batch_size == 0:
        micro_bs = find_max_batch_size(model, train_dataset, device)
    else:
        micro_bs = args.micro_batch_size
    
    grad_accum_steps = args.effective_batch_size // micro_bs
    print(f"⚙️ Config: Micro Batch={micro_bs}, Accum Steps={grad_accum_steps} => Effective={micro_bs*grad_accum_steps}")

    train_loader = DataLoader(train_dataset, batch_size=micro_bs, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=micro_bs, shuffle=False, num_workers=args.num_workers)

    # 4. Optimizer & Loss
    # Use 8-bit AdamW
    optimizer = bnb.optim.AdamW8bit(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr, 
        weight_decay=args.wd
    )
    
    # Warmup Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)
    
    scaler = GradScaler()
    early_stopper = EarlyStopper(patience=2)

    # 5. Resume Checkpoint Logic
    start_epoch = 0
    checkpoint_name = f"{args.scenario}_checkpoint.pt"
    os.makedirs(args.output_dir, exist_ok=True)
    local_checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
    if args.resume_from_checkpoint:
        if s3_mgr.download_checkpoint(f"{TrainingRunConfig.CHECKPOINTS_PREFIX}/{checkpoint_name}", local_checkpoint_path):
            checkpoint = torch.load(local_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"⏩ Resuming from epoch {start_epoch}")

    # 6. Training Loop
    print("🚀 Starting Training...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            # Mixed Precision Forward
            with autocast():
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, return_loss=True)
                loss = outputs.loss / grad_accum_steps # Scale loss

            # Backward
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:
                # Gradient Clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
                wandb.log({"train_loss": loss.item() * grad_accum_steps, "lr": scheduler.get_last_lr()[0]})

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, return_loss=True)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f}")
        wandb.log({"val_loss": avg_val_loss, "epoch": epoch})

        # Save Checkpoint (Local then S3)
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss
        }
        torch.save(ckpt, local_checkpoint_path)
        s3_mgr.upload_checkpoint(local_checkpoint_path, f"{TrainingRunConfig.CHECKPOINTS_PREFIX}/{checkpoint_name}")

        # Early Stopping
        if early_stopper.early_stop(avg_val_loss):
            print("🛑 Early Stopping triggered")
            break
            
    wandb.finish()

if __name__ == "__main__":
    main()