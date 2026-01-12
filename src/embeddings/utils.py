"""
Utility functions for loading CLIP models.

This module provides helpers for loading both base and fine-tuned CLIP models
for embedding generation.
"""

from pathlib import Path
from typing import Optional

import torch
from transformers import CLIPModel, CLIPProcessor

from src.config import ModelConfig, ModelPaths


def load_base_model(device: str = "cuda") -> tuple[CLIPModel, CLIPProcessor]:
    """
    Load the base (pre-trained) CLIP model.
    
    Args:
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading base CLIP model: {ModelConfig.MODEL_NAME}")
    model = CLIPModel.from_pretrained(ModelConfig.MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(ModelConfig.MODEL_NAME)
    
    model.to(device)
    model.eval()
    
    return model, processor


def load_finetuned_model(
    model_path: Optional[Path] = None,
    device: str = "cuda"
) -> tuple[Optional[CLIPModel], Optional[CLIPProcessor]]:
    """
    Load the fine-tuned CLIP model.
    
    Args:
        model_path: Path to fine-tuned model directory.
                   If None, uses default path from config.
        device: Device to load model on
        
    Returns:
        Tuple of (model, processor), or (None, None) if model not found
    """
    if model_path is None:
        model_path = ModelPaths.FINETUNED_DIR
    
    if not model_path.exists() or not (model_path / "config.json").exists():
        print(f"Fine-tuned model not found at {model_path}")
        print("Returning None. Will use base model only.")
        return None, None
    
    print(f"Loading fine-tuned CLIP model from: {model_path}")
    
    try:
        model = CLIPModel.from_pretrained(str(model_path))
        processor = CLIPProcessor.from_pretrained(str(model_path))
        
        model.to(device)
        model.eval()
        
        return model, processor
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        return None, None


def get_model_embedding_dim(model: CLIPModel) -> int:
    """
    Get the embedding dimension of a CLIP model.
    
    Args:
        model: CLIP model instance
        
    Returns:
        Embedding dimension
    """
    # CLIP models have config with projection_dim
    return model.config.projection_dim

