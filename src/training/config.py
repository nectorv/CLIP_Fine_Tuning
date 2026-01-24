import argparse

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