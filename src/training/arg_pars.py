import argparse

from src.config import TrainingRunConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on Furniture Dataset")

    # --- Infrastructure & Data ---
    parser.add_argument("--data_dir", type=str, default=TrainingRunConfig.DATA_DIR, help="Local path for data")
    parser.add_argument("--s3_bucket", type=str, default=TrainingRunConfig.S3_BUCKET, help="S3 Bucket name")
    parser.add_argument("--s3_prefix", type=str, default=TrainingRunConfig.S3_PREFIX, help="S3 folder path")
    parser.add_argument("--output_dir", type=str, default=TrainingRunConfig.OUTPUT_DIR)
    parser.add_argument("--mini_train", action="store_true", help="Use only a subset of data for testing")

    # --- Training Setup ---
    parser.add_argument("--scenario", type=str, default=TrainingRunConfig.DEFAULT_SCENARIO, 
                        choices=TrainingRunConfig.SCENARIO_CHOICES,
                        help="Ablation study scenario")
    parser.add_argument("--epochs", type=int, default=TrainingRunConfig.EPOCHS)
    parser.add_argument("--effective_batch_size", type=int, default=TrainingRunConfig.EFFECTIVE_BATCH_SIZE, help="Target accumulated batch size")
    parser.add_argument("--micro_batch_size", type=int, default=TrainingRunConfig.MICRO_BATCH_SIZE, help="Physical batch size (0 = auto-find)")
    parser.add_argument("--num_workers", type=int, default=TrainingRunConfig.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=TrainingRunConfig.SEED)

    # --- Optimization ---
    parser.add_argument("--lr", type=float, default=TrainingRunConfig.LR)
    parser.add_argument("--wd", type=float, default=TrainingRunConfig.WEIGHT_DECAY, help="Weight Decay")
    parser.add_argument("--warmup_steps", type=int, default=TrainingRunConfig.WARMUP_STEPS)
    parser.add_argument("--grad_clip", type=float, default=TrainingRunConfig.GRAD_CLIP)
    parser.add_argument("--augmentation", type=str, default=TrainingRunConfig.AUGMENTATION_DEFAULT, choices=TrainingRunConfig.AUGMENTATION_CHOICES, help="simple=Flip+Crop, advanced=Trivial+Erase")

    # --- LoRA Specifics ---
    parser.add_argument("--lora_r", type=int, default=TrainingRunConfig.LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=TrainingRunConfig.LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=TrainingRunConfig.LORA_DROPOUT)

    # --- WandB & Resume ---
    parser.add_argument("--wandb_project", type=str, default=TrainingRunConfig.WANDB_PROJECT)
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from S3 if available")

    args = parser.parse_args()
    return args