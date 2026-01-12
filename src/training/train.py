"""
Training script for CLIP fine-tuning on furniture data.

This module implements the complete training pipeline with:
- Freeze/unfreeze strategy for gradual fine-tuning
- Validation monitoring
- Checkpointing and best model saving
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src.config import (
    DataPaths,
    ModelConfig,
    ModelPaths,
    TrainingConfig,
)
from src.training.dataset import create_dataloaders
from src.training.loss import ContrastiveLoss


class CLIPTrainer:
    """
    Trainer for fine-tuning CLIP models on furniture data.
    
    Implements a two-phase training strategy:
    1. Freeze backbones, train only projection layers
    2. Unfreeze and fine-tune with lower learning rate
    """
    
    def __init__(self):
        """Initialize the trainer with model, loss, and optimizers."""
        self.device = torch.device(TrainingConfig.DEVICE)
        print(f"Using device: {self.device}")
        
        # Load model and processor
        print(f"Loading model: {ModelConfig.MODEL_NAME}")
        self.model = CLIPModel.from_pretrained(ModelConfig.MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(ModelConfig.MODEL_NAME)
        self.model.to(self.device)
        
        # Initialize loss
        self.criterion = ContrastiveLoss(temperature=0.07)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []
        
        # Create directories
        ModelPaths.ensure_dirs()
    
    def freeze_backbones(self):
        """Freeze vision and text encoders, keep only projection layers trainable."""
        print("Freezing vision and text encoders...")
        
        # Freeze vision encoder
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
        
        # Freeze text encoder
        for param in self.model.text_model.parameters():
            param.requires_grad = False
        
        # Keep projection layers trainable
        for param in self.model.visual_projection.parameters():
            param.requires_grad = True
        for param in self.model.text_projection.parameters():
            param.requires_grad = True
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    
    def unfreeze_backbones(self):
        """Unfreeze all parameters for full fine-tuning."""
        print("Unfreezing all parameters...")
        
        for param in self.model.parameters():
            param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    
    def create_optimizer(self, learning_rate: float):
        """
        Create optimizer for current trainable parameters.
        
        Args:
            learning_rate: Learning rate for optimization
            
        Returns:
            AdamW optimizer
        """
        # Get only trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=TrainingConfig.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    def create_scheduler(self, optimizer, num_training_steps: int, warmup_steps: int):
        """
        Create learning rate scheduler with warmup.
        
        Args:
            optimizer: Optimizer instance
            num_training_steps: Total number of training steps
            warmup_steps: Number of warmup steps
            
        Returns:
            Learning rate scheduler
        """
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=1e-7
        )
        
        # Sequential scheduler (warmup then cosine)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        return scheduler
    
    def train_epoch(self, train_loader, optimizer, scheduler, epoch: int):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training dataloader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Forward pass
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get embeddings
            image_embeddings = outputs.image_embeds
            text_embeddings = outputs.text_embeds
            
            # Compute loss
            loss = self.criterion(image_embeddings, text_embeddings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: Validation dataloader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for batch in pbar:
                # Move to device
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get embeddings
                image_embeddings = outputs.image_embeds
                text_embeddings = outputs.text_embeds
                
                # Compute loss
                loss = self.criterion(image_embeddings, text_embeddings)
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: float, optimizer, scheduler):
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Current validation loss
            optimizer: Optimizer state
            scheduler: Scheduler state
        """
        checkpoint_path = ModelPaths.CHECKPOINTS_DIR / f"checkpoint_epoch_{epoch+1}.pt"
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def save_best_model(self):
        """Save the best model based on validation loss."""
        model_path = ModelPaths.FINETUNED_DIR
        self.model.save_pretrained(model_path)
        self.processor.save_pretrained(model_path)
        print(f"Best model saved to: {model_path}")
    
    def train(
        self,
        train_loader,
        val_loader,
        freeze_epochs: int = 1,
        total_epochs: int = 5
    ):
        """
        Main training loop with freeze/unfreeze strategy.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            freeze_epochs: Number of epochs with frozen backbones
            total_epochs: Total number of epochs
        """
        print("\n" + "="*60)
        print("Starting CLIP Fine-Tuning")
        print("="*60)
        
        # Phase 1: Freeze backbones
        print(f"\nPhase 1: Training projection layers (Epochs 1-{freeze_epochs})")
        self.freeze_backbones()
        
        # Calculate training steps
        num_train_steps = len(train_loader) * freeze_epochs
        warmup_steps = min(TrainingConfig.WARMUP_STEPS, num_train_steps // 10)
        
        # Create optimizer and scheduler for frozen phase
        optimizer = self.create_optimizer(TrainingConfig.INITIAL_LR)
        scheduler = self.create_scheduler(optimizer, num_train_steps, warmup_steps)
        
        # Train frozen phase
        for epoch in range(freeze_epochs):
            print(f"\n--- Epoch {epoch+1}/{freeze_epochs} (Frozen) ---")
            
            train_loss = self.train_epoch(train_loader, optimizer, scheduler, epoch)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % TrainingConfig.SAVE_EVERY_N_EPOCHS == 0:
                self.save_checkpoint(epoch, val_loss, optimizer, scheduler)
            
            # Update best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_best_model()
        
        # Phase 2: Unfreeze and fine-tune
        if total_epochs > freeze_epochs:
            print(f"\nPhase 2: Full fine-tuning (Epochs {freeze_epochs+1}-{total_epochs})")
            self.unfreeze_backbones()
            
            # Recalculate steps for unfrozen phase
            remaining_epochs = total_epochs - freeze_epochs
            num_train_steps = len(train_loader) * remaining_epochs
            warmup_steps = min(500, num_train_steps // 10)
            
            # Create new optimizer with lower LR
            optimizer = self.create_optimizer(TrainingConfig.UNFROZEN_LR)
            scheduler = self.create_scheduler(optimizer, num_train_steps, warmup_steps)
            
            # Train unfrozen phase
            for epoch in range(freeze_epochs, total_epochs):
                print(f"\n--- Epoch {epoch+1}/{total_epochs} (Unfrozen) ---")
                
                train_loss = self.train_epoch(train_loader, optimizer, scheduler, epoch)
                val_loss = self.validate(val_loader)
                
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Save checkpoint
                if (epoch + 1) % TrainingConfig.SAVE_EVERY_N_EPOCHS == 0:
                    self.save_checkpoint(epoch, val_loss, optimizer, scheduler)
                
                # Update best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_best_model()
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print("="*60)


def main():
    """Main entry point for training script."""
    # Check if cleaned data exists
    if not DataPaths.CLEANED_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned CSV not found: {DataPaths.CLEANED_CSV_PATH}\n"
            "Please run the cleaning script first: python -m src.cleaning.cleaner"
        )
    
    # Check if image directory exists
    if not Path(DataPaths.IMAGE_DIR).exists():
        raise FileNotFoundError(f"Image directory not found: {DataPaths.IMAGE_DIR}")
    
    # Initialize trainer (this loads the model and processor)
    trainer = CLIPTrainer()
    
    # Create dataloaders with processor
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        csv_path=DataPaths.CLEANED_CSV_PATH,
        image_dir=Path(DataPaths.IMAGE_DIR),
        processor=trainer.processor,
        batch_size=TrainingConfig.BATCH_SIZE,
        val_split=TrainingConfig.VAL_SPLIT,
        num_workers=4,
        image_column="image_path",  # Update based on your CSV column name
        title_column="cleaned_title"
    )
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        freeze_epochs=TrainingConfig.FREEZE_EPOCHS,
        total_epochs=TrainingConfig.EPOCHS
    )


if __name__ == "__main__":
    main()

