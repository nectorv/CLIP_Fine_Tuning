import time
import math
import torch
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb
import os

from src.training.arg_pars import parse_args
from src.training.utils import S3Manager, EarlyStopper, find_max_batch_size
from src.training.data import FurnitureDataset, get_transforms, get_eval_transforms
from src.training.model import get_model
from src.config import TrainingRunConfig

def compute_batch_alignment(logits_per_image, logits_per_text):
    batch_size = logits_per_image.size(0)
    targets = torch.arange(batch_size, device=logits_per_image.device)
    img_to_text = (logits_per_image.argmax(dim=1) == targets).float().mean()
    text_to_img = (logits_per_text.argmax(dim=1) == targets).float().mean()
    batch_acc = (img_to_text + text_to_img) * 0.5
    return img_to_text, text_to_img, batch_acc

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup WandB (Resume logic)
    wandb.init(project=args.wandb_project, config=args, resume="allow", name=f"{args.scenario}_lr{args.lr}")

    # 2. Data Prep
    s3_mgr = S3Manager(args.s3_bucket)
    s3_mgr.download_dataset(args.s3_prefix, args.data_dir)
    
    transform = get_transforms(args.augmentation)
    eval_transform = get_eval_transforms()
    model, processor = get_model(args.scenario, args.lora_r)
    model.to(device)
    param_device = next(model.parameters()).device
    print(f"🧠 Model loaded on device: {param_device}")

    train_dir = os.path.join(args.data_dir, "mini_train" if args.mini_train else "train")
    val_dir = os.path.join(args.data_dir, "validation")

    test_dir = os.path.join(args.data_dir, "test")

    train_dataset = FurnitureDataset(train_dir, processor, transform=transform)
    val_dataset = FurnitureDataset(val_dir, processor, transform=eval_transform)
    test_dataset = FurnitureDataset(test_dir, processor, transform=eval_transform)

    if args.sanity_check_samples > 0:
        sample_count = args.sanity_check_samples
        print("🔎 Sanity check samples:")
        for split_name, dataset in ("train", train_dataset), ("val", val_dataset), ("test", test_dataset):
            split_size = len(dataset)
            take = min(sample_count, split_size)
            if take == 0:
                print(f"  {split_name}: empty")
                continue
            print(f"  {split_name}: showing {take}/{split_size}")
            for idx in range(take):
                caption = dataset.captions[idx]
                token_len = int(dataset[idx]["input_ids"].numel())
                print(f"    [{idx}] tokens={token_len} caption=\"{caption[:160]}\"")

    if len(train_dataset) == 0:
        raise ValueError(
            f"No images found in {train_dir}. Check --s3_bucket, --s3_prefix, and --data_dir."
        )
    if len(val_dataset) == 0:
        raise ValueError(
            f"No images found in {val_dir}. Validation folder is required."
        )
    if len(test_dataset) == 0:
        raise ValueError(
            f"No images found in {test_dir}. Test folder is required for evaluation."
        )

    # 3. Dynamic Batch Size
    if args.micro_batch_size == 0:
        micro_bs = find_max_batch_size(model, train_dataset, device)
    else:
        micro_bs = args.micro_batch_size
    
    grad_accum_steps = args.effective_batch_size // micro_bs
    if grad_accum_steps < 1:
        raise ValueError(
            f"effective_batch_size ({args.effective_batch_size}) must be >= micro_batch_size ({micro_bs})."
        )
    print(f"⚙️ Config: Micro Batch={micro_bs}, Accum Steps={grad_accum_steps} => Effective={micro_bs*grad_accum_steps}")

    train_loader = DataLoader(train_dataset, batch_size=micro_bs, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=micro_bs, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=micro_bs, shuffle=False, num_workers=args.num_workers)

    # 4. Optimizer & Loss
    # Use 8-bit AdamW
    optimizer = bnb.optim.AdamW8bit(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr, 
        weight_decay=args.wd
    )
    
    # Warmup Scheduler
    updates_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = updates_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)
    
    scaler = GradScaler()
    early_stopper = EarlyStopper(patience=2)

    # 5. Checkpoint Setup
    start_epoch = 0
    checkpoint_name = f"{args.scenario}_checkpoint.pt"
    os.makedirs(args.output_dir, exist_ok=True)
    local_checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
    if args.enable_s3_checkpoints and args.resume_from_checkpoint:
        s3_checkpoint_path = f"{TrainingRunConfig.CHECKPOINTS_PREFIX}/{checkpoint_name}"
        if s3_mgr.download_checkpoint(s3_checkpoint_path, local_checkpoint_path):
            checkpoint = torch.load(local_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"⏩ Resuming from epoch {start_epoch}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔥 Total Trainable Params: {trainable_params}")

    # 6. Training Loop
    print("🚀 Starting Training...")
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0
        
        accum_steps = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Mixed Precision Forward
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    return_loss=True
                )
                loss = outputs.loss / grad_accum_steps # Scale loss
            with torch.no_grad():
                img_to_text_acc, text_to_img_acc, batch_acc = compute_batch_alignment(
                    outputs.logits_per_image,
                    outputs.logits_per_text
                )

            # Backward
            scaler.scale(loss).backward()
            accum_steps += 1

            if accum_steps == grad_accum_steps or (step + 1) == len(train_loader):
                # Gradient Clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                accum_steps = 0
                
                wandb.log({
                    "train_loss": loss.item() * grad_accum_steps,
                    "lr": scheduler.get_last_lr()[0],
                    "batch_acc": batch_acc.item(),
                    "batch_acc_img_to_text": img_to_text_acc.item(),
                    "batch_acc_text_to_img": text_to_img_acc.item()
                })

        # Validation
        model.eval()
        val_loss = 0
        val_img_to_text = 0.0
        val_text_to_img = 0.0
        val_batch_acc = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    return_loss=True
                )
                val_loss += outputs.loss.item()
                img_to_text_acc, text_to_img_acc, batch_acc = compute_batch_alignment(
                    outputs.logits_per_image,
                    outputs.logits_per_text
                )
                val_img_to_text += img_to_text_acc.item()
                val_text_to_img += text_to_img_acc.item()
                val_batch_acc += batch_acc.item()
                val_steps += 1
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_img_to_text = val_img_to_text / max(1, val_steps)
        avg_val_text_to_img = val_text_to_img / max(1, val_steps)
        avg_val_batch_acc = val_batch_acc / max(1, val_steps)
        epoch_duration = time.perf_counter() - epoch_start
        print(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_duration:.2f}s")
        wandb.log({
            "val_loss": avg_val_loss,
            "val_batch_acc": avg_val_batch_acc,
            "val_batch_acc_img_to_text": avg_val_img_to_text,
            "val_batch_acc_text_to_img": avg_val_text_to_img,
            "epoch": epoch
        })

        # Save Checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss
        }
        torch.save(ckpt, local_checkpoint_path)
        if args.enable_s3_checkpoints:
            s3_checkpoint_path = f"{TrainingRunConfig.CHECKPOINTS_PREFIX}/{checkpoint_name}"
            s3_mgr.upload_checkpoint(local_checkpoint_path, s3_checkpoint_path)

        # Early Stopping
        if early_stopper.early_stop(avg_val_loss):
            print("🛑 Early Stopping triggered")
            break

    # 7. Final Test Evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_loss=True
            )
            test_loss += outputs.loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    wandb.log({"test_loss": avg_test_loss})

    wandb.finish()

if __name__ == "__main__":
    main()