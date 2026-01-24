import torch
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb
import os
import glob

from config import parse_args
from utils import S3Manager, EarlyStopper, find_max_batch_size
from data import get_wds_pipeline, get_transforms
from model import get_model

def count_samples(folder_path):
    """Estimation rapide ou précise du nombre d'images (nécessaire pour WebDataset + Tqdm)"""
    # Si vous avez un fichier metadata.json c'est mieux, sinon on estime
    # Pour l'instant, on met une valeur par défaut ou on compte si on a le temps
    # Pour WebDataset, souvent on hardcode ou on passe l'info en argument
    return 50000 # Valeur par défaut pour votre dataset

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup WandB
    wandb.init(project=args.wandb_project, config=args, resume="allow", name=f"{args.scenario}_lr{args.lr}")

    # 2. Data Prep (Téléchargement Local depuis S3)
    s3_mgr = S3Manager(args.s3_bucket)
    
    # Définition des dossiers locaux cibles
    local_train = os.path.join(args.data_dir, "mini_train" if args.mini_train else "train")
    local_val = os.path.join(args.data_dir, "validation")
    
    # Téléchargement (On suppose que s3_prefix contient les sous-dossiers train/validation)
    # Exemple S3 structure: s3://bucket/data/train/shard-000.tar
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

    train_dataset = get_wds_pipeline(
        local_train, processor, transform=transform, is_train=True, epoch_len=train_len
    )
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
        micro_bs = args.micro_batch_size
    
    grad_accum_steps = args.effective_batch_size // micro_bs

    # DataLoader standard de PyTorch fonctionne très bien avec l'objet WebDataset
    train_loader = DataLoader(train_dataset, batch_size=micro_bs, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=micro_bs, num_workers=args.num_workers)
    
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
    if args.resume_from_checkpoint:
        if s3_mgr.download_checkpoint(f"checkpoints/{checkpoint_name}", checkpoint_name):
            checkpoint = torch.load(checkpoint_name)
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
        torch.save(ckpt, checkpoint_name)
        s3_mgr.upload_checkpoint(checkpoint_name, f"checkpoints/{checkpoint_name}")

        # Early Stopping
        if early_stopper.early_stop(avg_val_loss):
            print("🛑 Early Stopping triggered")
            break
            
    wandb.finish()

if __name__ == "__main__":
    main()