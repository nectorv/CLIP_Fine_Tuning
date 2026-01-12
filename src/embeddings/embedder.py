"""
Embedding generation script for furniture semantic search.

This module generates embeddings for all images and texts in the dataset
using both base and fine-tuned CLIP models, saving them to a Parquet file.
"""

import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPProcessor

from src.config import DataPaths, EmbeddingConfig, ModelConfig
from src.embeddings.utils import load_base_model, load_finetuned_model, get_model_embedding_dim


class EmbeddingGenerator:
    """
    Generator for CLIP embeddings.
    
    Processes the full dataset and generates embeddings using both
    base and fine-tuned CLIP models.
    """
    
    def __init__(self):
        """Initialize the embedding generator."""
        self.device = torch.device(EmbeddingConfig.DEVICE)
        print(f"Using device: {self.device}")
        
        # Load models
        print("\nLoading models...")
        self.base_model, self.base_processor = load_base_model(self.device)
        self.finetuned_model, self.finetuned_processor = load_finetuned_model(device=self.device)
        
        # Get embedding dimensions
        self.base_dim = get_model_embedding_dim(self.base_model)
        if self.finetuned_model:
            self.finetuned_dim = get_model_embedding_dim(self.finetuned_model)
        else:
            self.finetuned_dim = None
        
        print(f"Base model embedding dim: {self.base_dim}")
        if self.finetuned_dim:
            print(f"Fine-tuned model embedding dim: {self.finetuned_dim}")
    
    def generate_image_embeddings(
        self,
        model: torch.nn.Module,
        processor: CLIPProcessor,
        image_paths: list[Path],
        batch_size: int = 128
    ) -> torch.Tensor:
        """
        Generate image embeddings for a list of images.
        
        Args:
            model: CLIP model
            processor: CLIP processor
            image_paths: List of paths to images
            batch_size: Batch size for processing
            
        Returns:
            Tensor of embeddings [num_images, embedding_dim]
        """
        from PIL import Image
        
        embeddings = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Generating image embeddings"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            # Load images
            for img_path in batch_paths:
                try:
                    if img_path.exists():
                        img = Image.open(img_path).convert("RGB")
                        batch_images.append(img)
                    else:
                        # Create black placeholder
                        batch_images.append(Image.new("RGB", (224, 224), color="black"))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    batch_images.append(Image.new("RGB", (224, 224), color="black"))
            
            # Process batch
            inputs = processor(images=batch_images, return_tensors="pt", padding=True)
            pixel_values = inputs["pixel_values"].to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                image_features = model.get_image_features(pixel_values=pixel_values)
                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            embeddings.append(image_features.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def generate_text_embeddings(
        self,
        model: torch.nn.Module,
        processor: CLIPProcessor,
        texts: list[str],
        batch_size: int = 128
    ) -> torch.Tensor:
        """
        Generate text embeddings for a list of texts.
        
        Args:
            model: CLIP model
            processor: CLIP processor
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            Tensor of embeddings [num_texts, embedding_dim]
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating text embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Process batch
            inputs = processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=ModelConfig.MAX_TEXT_LENGTH
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                text_features = model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # Normalize
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            embeddings.append(text_features.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def generate_late_fusion_embeddings(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate late fusion embeddings by averaging image and text embeddings.
        
        Args:
            image_embeddings: Image embeddings [N, dim]
            text_embeddings: Text embeddings [N, dim]
            
        Returns:
            Fused embeddings [N, dim]
        """
        # Average the embeddings
        fused = (image_embeddings + text_embeddings) / 2.0
        # Renormalize
        fused = fused / fused.norm(dim=-1, keepdim=True)
        return fused
    
    def process_dataset(self, csv_path: Path, image_dir: Path) -> pd.DataFrame:
        """
        Process the full dataset and generate embeddings.
        
        Args:
            csv_path: Path to cleaned CSV file
            image_dir: Directory containing images
            
        Returns:
            DataFrame with embeddings added
        """
        print("\nLoading dataset...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows")
        
        # Get image paths and texts
        image_paths = []
        texts = []
        
        # Try to detect image column name
        image_column = None
        for col in ["image_path", "image", "filename", "file_name", "path"]:
            if col in df.columns:
                image_column = col
                break
        
        if image_column is None:
            print("Warning: No image path column found. Using index-based filenames.")
        
        for idx, row in df.iterrows():
            # Get image path
            if image_column and image_column in row.index:
                img_name = row[image_column]
            else:
                # Fallback: use index
                img_name = f"{idx}.jpg"
            
            if pd.isna(img_name):
                img_name = f"{idx}.jpg"
            
            img_path = Path(image_dir) / str(img_name) if not Path(str(img_name)).is_absolute() else Path(str(img_name))
            image_paths.append(img_path)
            
            # Get text
            text = str(row.get("cleaned_title", "")).strip()
            if not text:
                text = "furniture item"
            texts.append(text)
        
        # Generate base model embeddings
        print("\n" + "="*60)
        print("Generating Base CLIP Embeddings")
        print("="*60)
        
        base_img_emb = self.generate_image_embeddings(
            self.base_model,
            self.base_processor,
            image_paths,
            batch_size=EmbeddingConfig.BATCH_SIZE
        )
        
        base_text_emb = self.generate_text_embeddings(
            self.base_model,
            self.base_processor,
            texts,
            batch_size=EmbeddingConfig.BATCH_SIZE
        )
        
        base_fused = self.generate_late_fusion_embeddings(base_img_emb, base_text_emb)
        
        # Generate fine-tuned model embeddings (if available)
        if self.finetuned_model:
            print("\n" + "="*60)
            print("Generating Fine-Tuned CLIP Embeddings")
            print("="*60)
            
            finetuned_img_emb = self.generate_image_embeddings(
                self.finetuned_model,
                self.finetuned_processor,
                image_paths,
                batch_size=EmbeddingConfig.BATCH_SIZE
            )
            
            finetuned_text_emb = self.generate_text_embeddings(
                self.finetuned_model,
                self.finetuned_processor,
                texts,
                batch_size=EmbeddingConfig.BATCH_SIZE
            )
            
            finetuned_fused = self.generate_late_fusion_embeddings(
                finetuned_img_emb,
                finetuned_text_emb
            )
        else:
            finetuned_fused = None
        
        # Convert to lists for DataFrame storage
        print("\nConverting embeddings to lists...")
        df["vec_clip_base"] = base_fused.tolist()
        
        if finetuned_fused is not None:
            df["vec_clip_finetuned"] = finetuned_fused.tolist()
        else:
            df["vec_clip_finetuned"] = None
        
        return df
    
    def save_embeddings(self, df: pd.DataFrame, output_path: Path):
        """
        Save embeddings to Parquet file.
        
        Args:
            df: DataFrame with embeddings
            output_path: Path to save Parquet file
        """
        print(f"\nSaving embeddings to: {output_path}")
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet
        df.to_parquet(output_path, index=False, engine="pyarrow")
        
        print(f"Saved {len(df)} embeddings")
        print(f"Columns: {df.columns.tolist()}")


def main():
    """Main entry point for embedding generation."""
    # Check if cleaned data exists
    if not DataPaths.CLEANED_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned CSV not found: {DataPaths.CLEANED_CSV_PATH}\n"
            "Please run the cleaning script first: python -m src.cleaning.cleaner"
        )
    
    # Check if image directory exists
    image_dir = Path(DataPaths.IMAGE_DIR)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Initialize generator
    generator = EmbeddingGenerator()
    
    # Process dataset
    df = generator.process_dataset(
        csv_path=DataPaths.CLEANED_CSV_PATH,
        image_dir=image_dir
    )
    
    # Save embeddings
    generator.save_embeddings(df, DataPaths.MASTER_VECTORS_PATH)
    
    print("\n" + "="*60)
    print("Embedding Generation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

