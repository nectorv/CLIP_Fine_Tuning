"""
Contrastive loss functions for CLIP fine-tuning.

This module implements InfoNCE (NT-Xent) loss, which is the standard
contrastive learning objective for CLIP models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    InfoNCE (NT-Xent) contrastive loss for CLIP.
    
    This loss maximizes the similarity between matching image-text pairs
    while minimizing similarity between non-matching pairs within a batch.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize the contrastive loss.
        
        Args:
            temperature: Temperature parameter for scaling logits.
                        Lower values make the distribution sharper.
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self, 
        image_embeddings: torch.Tensor, 
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss between image and text embeddings.
        
        Args:
            image_embeddings: Image embeddings [batch_size, embedding_dim]
            text_embeddings: Text embeddings [batch_size, embedding_dim]
            
        Returns:
            Scalar loss value
        """
        batch_size = image_embeddings.size(0)
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        
        # Compute similarity matrix
        # Shape: [batch_size, batch_size]
        logits = torch.matmul(image_embeddings, text_embeddings.t()) / self.temperature
        
        # Create labels (diagonal elements are positive pairs)
        labels = torch.arange(batch_size, device=image_embeddings.device)
        
        # Compute loss in both directions (image-to-text and text-to-image)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        # Average the two directions
        loss = (loss_i2t + loss_t2i) / 2.0
        
        return loss


class SymmetricContrastiveLoss(nn.Module):
    """
    Symmetric contrastive loss with optional hard negative mining.
    
    This is an enhanced version that can handle hard negatives and
    provides more stable training.
    """
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.0):
        """
        Initialize symmetric contrastive loss.
        
        Args:
            temperature: Temperature parameter for scaling
            margin: Optional margin for hard negative mining (0 = standard InfoNCE)
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute symmetric contrastive loss.
        
        Args:
            image_embeddings: Image embeddings [batch_size, embedding_dim]
            text_embeddings: Text embeddings [batch_size, embedding_dim]
            
        Returns:
            Scalar loss value
        """
        batch_size = image_embeddings.size(0)
        
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_embeddings, text_embeddings.t()) / self.temperature
        
        # Create labels
        labels = torch.arange(batch_size, device=image_embeddings.device)
        
        # Apply margin if specified
        if self.margin > 0:
            # Subtract margin from positive pairs
            logits = logits - self.margin * torch.eye(
                batch_size, 
                device=image_embeddings.device
            )
        
        # Symmetric loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        loss = (loss_i2t + loss_t2i) / 2.0
        
        return loss

