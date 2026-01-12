"""
Custom PyTorch Dataset for CLIP fine-tuning.

This module implements a dataset that loads furniture images and their
corresponding cleaned titles for contrastive learning.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor

from src.config import DataPaths, ModelConfig


class FurnitureDataset(Dataset):
    """
    Dataset for furniture images and cleaned titles.
    
    Loads images from disk and tokenizes text using CLIP's processor.
    Handles missing images gracefully by skipping or using placeholder.
    """
    
    def __init__(
        self,
        csv_path: Path,
        image_dir: Path,
        processor: CLIPProcessor,
        image_column: str = "image_path",
        title_column: str = "cleaned_title",
        transform=None
    ):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV file with image paths and titles
            image_dir: Base directory for images
            processor: CLIP processor for tokenization
            image_column: Column name for image paths/filenames
            title_column: Column name for cleaned titles
            transform: Optional image transforms (processor handles this)
        """
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.image_column = image_column
        self.title_column = title_column
        self.transform = transform
        
        # Filter out rows with missing images or empty titles
        self._filter_valid_rows()
        
        print(f"Dataset initialized with {len(self.df)} valid samples")
    
    def _filter_valid_rows(self):
        """Remove rows with missing images or empty titles."""
        initial_count = len(self.df)
        
        # Check for valid titles
        valid_mask = (
            self.df[self.title_column].notna() &
            (self.df[self.title_column].astype(str).str.strip() != "")
        )
        
        # Check for existing images
        if self.image_column in self.df.columns:
            def image_exists(row):
                img_path = self._get_image_path(row)
                return img_path.exists() if img_path else False
            
            image_mask = self.df.apply(image_exists, axis=1)
            valid_mask = valid_mask & image_mask
        
        self.df = self.df[valid_mask].reset_index(drop=True)
        
        removed = initial_count - len(self.df)
        if removed > 0:
            print(f"Removed {removed} rows with missing images or empty titles")
    
    def _get_image_path(self, row) -> Optional[Path]:
        """
        Get the full path to an image from a row.
        
        Args:
            row: DataFrame row
            
        Returns:
            Path to image file, or None if not found
        """
        if self.image_column in row.index:
            img_name = row[self.image_column]
        else:
            # Try to infer from index or other columns
            img_name = f"{row.name}.jpg"  # Fallback
        
        if pd.isna(img_name):
            return None
        
        # Handle both relative and absolute paths
        img_path = Path(str(img_name))
        if not img_path.is_absolute():
            img_path = self.image_dir / img_path
        
        return img_path
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'image' and 'text' keys, processed by CLIP processor
        """
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self._get_image_path(row)
        if img_path is None or not img_path.exists():
            # Return a placeholder black image
            image = Image.new("RGB", (ModelConfig.IMAGE_SIZE, ModelConfig.IMAGE_SIZE), color="black")
        else:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                image = Image.new("RGB", (ModelConfig.IMAGE_SIZE, ModelConfig.IMAGE_SIZE), color="black")
        
        # Get text
        text = str(row[self.title_column]).strip()
        if not text:
            text = "furniture item"  # Fallback
        
        # Apply optional transform
        if self.transform:
            image = self.transform(image)
        
        # Process with CLIP processor
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Extract tensors (processor returns batched, so we squeeze)
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "text": text,  # Keep original text for logging
            "image_path": str(img_path) if img_path else ""
        }


def create_dataloaders(
    csv_path: Path,
    image_dir: Path,
    processor: CLIPProcessor,
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 4,
    image_column: str = "image_path",
    title_column: str = "cleaned_title"
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        csv_path: Path to cleaned CSV
        image_dir: Directory containing images
        processor: CLIP processor
        batch_size: Batch size for training
        val_split: Fraction of data for validation
        num_workers: Number of worker processes
        image_column: Column name for image paths
        title_column: Column name for cleaned titles
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create full dataset
    full_dataset = FurnitureDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        processor=processor,
        image_column=image_column,
        title_column=title_column
    )
    
    # Split into train/val
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # For consistent batch sizes
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batches for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader

