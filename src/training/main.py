import torch
import wandb
import os

from src.training.arg_pars import parse_args
from src.training.utils import S3Manager
from src.training.data import FurnitureDataset, get_transforms, get_eval_transforms
from src.training.model import get_model
from src.training.trainer import train_and_evaluate

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

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔥 Total Trainable Params: {trainable_params}")

    train_and_evaluate(
        args,
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        device,
        s3_mgr
    )

    wandb.finish()

if __name__ == "__main__":
    main()