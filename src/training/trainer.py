import math
import time

import bitsandbytes as bnb
import torch
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.training.checkpointing import load_checkpoint_if_available, save_checkpoint
from src.training.metrics import compute_batch_alignment
from src.training.utils import EarlyStopper, find_max_batch_size


def build_dataloaders(args, train_dataset, val_dataset, test_dataset, micro_bs):
    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_bs,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=micro_bs,
        shuffle=False,
        num_workers=args.num_workers
    )
    return train_loader, val_loader, test_loader


def setup_optimizer_and_scheduler(args, model, train_loader, grad_accum_steps):
    optimizer = bnb.optim.AdamW8bit(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.wd
    )
    updates_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = updates_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1
    )
    return optimizer, scheduler


def train_and_evaluate(args, model, train_dataset, val_dataset, test_dataset, device, s3_mgr):
    if args.micro_batch_size == 0:
        micro_bs = find_max_batch_size(model, train_dataset, device)
    else:
        micro_bs = args.micro_batch_size

    grad_accum_steps = args.effective_batch_size // micro_bs
    if grad_accum_steps < 1:
        raise ValueError(
            f"effective_batch_size ({args.effective_batch_size}) must be >= micro_batch_size ({micro_bs})."
        )
    print(
        f"⚙️ Config: Micro Batch={micro_bs}, Accum Steps={grad_accum_steps} => Effective={micro_bs * grad_accum_steps}"
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        args,
        train_dataset,
        val_dataset,
        test_dataset,
        micro_bs
    )

    optimizer, scheduler = setup_optimizer_and_scheduler(args, model, train_loader, grad_accum_steps)
    scaler = GradScaler()
    early_stopper = EarlyStopper(patience=args.patience)
    start_epoch = load_checkpoint_if_available(args, s3_mgr, model, optimizer)

    print("🚀 Starting Training...")
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        accum_steps = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    return_loss=True
                )
                loss = outputs.loss / grad_accum_steps
            with torch.no_grad():
                _, _, batch_acc = compute_batch_alignment(
                    outputs.logits_per_image,
                    outputs.logits_per_text
                )

            scaler.scale(loss).backward()
            accum_steps += 1

            if accum_steps == grad_accum_steps or (step + 1) == len(train_loader):
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
                    "batch_acc": batch_acc.item()
                })

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

        save_checkpoint(args, s3_mgr, model, optimizer, epoch, avg_val_loss)

        if early_stopper.early_stop(avg_val_loss):
            print("🛑 Early Stopping triggered")
            break

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
