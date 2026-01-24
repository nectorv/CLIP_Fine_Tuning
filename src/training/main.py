import torch
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb
import os
import glob
import argparse

from utils import S3Manager, EarlyStopper, find_max_batch_size
from data import get_wds_pipeline, get_transforms
from model import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on Furniture Dataset")

    # --- Infrastructure & Data ---
    parser.add_argument("--data_dir", type=str, default="/tmp/furniture_data", help="Local path for data")
    parser.add_argument("--s3_bucket", type=str, required=True, help="S3 Bucket name")
    parser.add_argument("--s3_prefix", type=str, default="data/train", help="S3 folder path")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--mini_train", action="store_true", help="Use only a subset of data for testing")

    # --- Training Setup ---
    parser.add_argument("--scenario", type=str, default="dual_lora", 
                        choices=["zero_shot", "linear_probe", "dual_lora", "vision_lora", "text_lora"],
                        help="Ablation study scenario")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--effective_batch_size", type=int, default=1024, help="Target accumulated batch size")
    parser.add_argument("--micro_batch_size", type=int, default=0, help="Physical batch size (0 = auto-find)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # --- Optimization ---
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.1, help="Weight Decay")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--augmentation", type=str, default="simple", choices=["simple", "advanced"], help="simple=Flip+Crop, advanced=Trivial+Erase")

    # --- LoRA Specifics ---
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # --- WandB & Resume ---
    parser.add_argument("--wandb_project", type=str, default="clip-furniture-finetune")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from S3 if available")

    args = parser.parse_args()
    return args

def count_samples(folder_path):
    """Estimate number of samples in the dataset."""
    return 50000

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    wandb.init(project=args.wandb_project, config=args, resume="allow", name=f"{args.scenario}_lr{args.lr}")

    s3_mgr = S3Manager(args.s3_bucket)
    
    local_train = os.path.join(args.data_dir, "mini_train" if args.mini_train else "train")
    local_val = os.path.join(args.data_dir, "validation")
    
    s3_train_prefix = f"{args.s3_prefix}/{'mini_train' if args.mini_train else 'train'}"
    s3_val_prefix = f"{args.s3_prefix}/validation"
    
    s3_mgr.download_dataset(s3_train_prefix, local_train)
    s3_mgr.download_dataset(s3_val_prefix, local_val)
    
    transform = get_transforms(args.augmentation)
    model, processor = get_model(args.scenario, args.lora_r)
    model.to(device)

    # 3. Création des Pipelines WebDataset
    # On doit estimer la longueur pour tqdm. 
    # Mini-train = 1000 (environ), Full = 50000.
    train_len = 1000 if args.mini_train else 45000 # Ajustez selon votre vrai split
    val_len = 500 if args.mini_train else 5000
train_len = 1000 if args.mini_train else 45000
    val_dataset = get_wds_pipeline(
        local_val, processor, transform=None, is_train=False, epoch_len=val_len
    )

    # 4. Dynamic Batch Size
    if args.micro_batch_size == 0:
        # Attention: find_max_batch_size doit être adapté pour WDS ou hardcodé pour commencer
        # Pour WDS, c'est plus compliqué de tester dynamiquement car c'est un stream.
        # Conseil Senior: Commencez avec 64 ou 128 hardcodé si vous changez de format.
        micro_bs = 64 
        print(f"⚠️ WDS Mode: Defaulting micro_batch to {micro_bs} (Batch Finder skipped)")
    else:
    if args.micro_batch_size == 0:
        micro_bs = 64 
        print(f"⚠️ WDS Mode: Defaulting micro_batch to {micro_bs} (Batch Finder skipped)")
    else:
        micro_bs = args.micro_batch_size
    
    grad_accum_steps = args.effective_batch_size // micro_bs

    train_loader = DataLoader(train_dataset, batch_size=micro_bs, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=micro_bs, num_workers=args.num_workers)
    
    optimizer = bnb.optim.AdamW8bit(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr, 
        weight_decay=args.wd
    )
    
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)
    
    scaler = GradScaler()
    early_stopper = EarlyStopper(patience=2)

    start_epoch = 0
    checkpoint_name = f"{args.scenario}_checkpoint.pt"
    if args.resume_from_checkpoint:
        if s3_mgr.download_checkpoint(f"checkpoints/{checkpoint_name}", checkpoint_name):
            checkpoint = torch.load(checkpoint_name)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"⏩ Resuming from epoch {start_epoch}")
es = batch["pixel_values"].to(device)

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
                
            with autocast():
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, return_loss=True)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:
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
        torch.save(ckpt, checkpoint_name)
        s3_mgr.upload_checkpoint(checkpoint_name, f"checkpoints/{checkpoint_name}")

        # Early Stopping
        if early_stopper.early_stop(avg_val_loss):
            print("🛑 Early Stopping triggered")
            break
            
    wandb.finish()

if __name__ == "__main__":
    main()