import os
import torch

from src.config import TrainingRunConfig


def get_checkpoint_paths(args):
    checkpoint_name = f"{args.scenario}_checkpoint.pt"
    os.makedirs(args.output_dir, exist_ok=True)
    local_checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
    s3_checkpoint_path = f"{TrainingRunConfig.CHECKPOINTS_PREFIX}/{checkpoint_name}"
    return checkpoint_name, local_checkpoint_path, s3_checkpoint_path


def load_checkpoint_if_available(args, s3_mgr, model, optimizer):
    start_epoch = 0
    if args.enable_s3_checkpoints and args.resume_from_checkpoint:
        _, local_checkpoint_path, s3_checkpoint_path = get_checkpoint_paths(args)
        if s3_mgr.download_checkpoint(s3_checkpoint_path, local_checkpoint_path):
            checkpoint = torch.load(local_checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"⏩ Resuming from epoch {start_epoch}")
    return start_epoch


def save_checkpoint(args, s3_mgr, model, optimizer, epoch, loss):
    _, local_checkpoint_path, s3_checkpoint_path = get_checkpoint_paths(args)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }
    torch.save(ckpt, local_checkpoint_path)
    if args.enable_s3_checkpoints:
        s3_mgr.upload_checkpoint(local_checkpoint_path, s3_checkpoint_path)
