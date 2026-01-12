"""
Arena GUI application for human-in-the-loop evaluation.

This module implements a matplotlib-based full-screen GUI for comparing
baseline and fine-tuned CLIP models side-by-side.
"""

import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor

from src.config import ArenaConfig, DataPaths, ModelConfig, ModelPaths
from src.arena.grading import ArenaGrader
from src.embeddings.utils import load_base_model, load_finetuned_model


class ArenaApp:
    """
    Full-screen GUI application for model comparison.
    
    Displays query image and top-K results from both baseline and
    fine-tuned models in a randomized, blind-test format.
    """
    
    def __init__(self):
        """Initialize the Arena application."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load models
        print("Loading models...")
        self.base_model, self.base_processor = load_base_model(self.device)
        self.finetuned_model, self.finetuned_processor = load_finetuned_model(device=self.device)
        
        if self.finetuned_model is None:
            raise RuntimeError(
                "Fine-tuned model not found. Please train the model first: "
                "python -m src.training.train"
            )
        
        # Load embeddings
        print("Loading embeddings...")
        if not DataPaths.MASTER_VECTORS_PATH.exists():
            raise FileNotFoundError(
                f"Embeddings not found: {DataPaths.MASTER_VECTORS_PATH}\n"
                "Please generate embeddings first: python -m src.embeddings.embedder"
            )
        
        self.embeddings_df = pd.read_parquet(DataPaths.MASTER_VECTORS_PATH)
        print(f"Loaded {len(self.embeddings_df)} embeddings")
        
        # Initialize grader
        self.grader = ArenaGrader()
        
        # Evaluation images
        self.eval_image_dir = Path(DataPaths.EVALUATION_IMAGE_DIR)
        if not self.eval_image_dir.exists():
            raise FileNotFoundError(
                f"Evaluation image directory not found: {self.eval_image_dir}"
            )
        
        # Get list of evaluation images
        self.eval_images = list(self.eval_image_dir.glob("*.jpg")) + \
                          list(self.eval_image_dir.glob("*.png")) + \
                          list(self.eval_image_dir.glob("*.jpeg"))
        
        if len(self.eval_images) == 0:
            raise FileNotFoundError(
                f"No evaluation images found in: {self.eval_image_dir}"
            )
        
        print(f"Found {len(self.eval_images)} evaluation images")
        
        # Matplotlib setup
        self.fig = None
        self.ax = None
        self.current_query = None
        self.baseline_is_row_a = None
    
    def get_image_embedding(self, image_path: Path, model, processor) -> torch.Tensor:
        """
        Get embedding for a single image.
        
        Args:
            image_path: Path to image
            model: CLIP model
            processor: CLIP processor
            
        Returns:
            Normalized embedding vector
        """
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        
        with torch.no_grad():
            embedding = model.get_image_features(pixel_values=pixel_values)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().squeeze(0)
    
    def get_text_embedding(self, text: str, model, processor) -> torch.Tensor:
        """
        Get embedding for a single text.
        
        Args:
            text: Text string
            model: CLIP model
            processor: CLIP processor
            
        Returns:
            Normalized embedding vector
        """
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            embedding = model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().squeeze(0)
    
    def search(
        self,
        query_embedding: torch.Tensor,
        method: str = "base",
        top_k: int = 4
    ) -> list[dict]:
        """
        Search for similar items using embeddings.
        
        Args:
            query_embedding: Query embedding vector
            method: "base" or "finetuned"
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with image paths and metadata
        """
        # Get embedding column
        if method == "base":
            emb_col = "vec_clip_base"
        else:
            emb_col = "vec_clip_finetuned"
        
        # Convert embeddings to tensor
        embeddings = torch.tensor(
            self.embeddings_df[emb_col].tolist(),
            dtype=torch.float32
        )
        
        # Compute similarities
        similarities = torch.matmul(query_embedding.unsqueeze(0), embeddings.t()).squeeze(0)
        
        # Get top-K indices
        top_indices = torch.topk(similarities, k=min(top_k, len(similarities))).indices.tolist()
        
        # Build results
        results = []
        for idx in top_indices:
            row = self.embeddings_df.iloc[idx]
            
            # Get image path
            img_path = row.get("image_path", "")
            if not img_path or pd.isna(img_path):
                # Try to construct from index
                img_path = Path(DataPaths.IMAGE_DIR) / f"{idx}.jpg"
            else:
                img_path = Path(img_path)
                if not img_path.is_absolute():
                    img_path = Path(DataPaths.IMAGE_DIR) / img_path
            
            results.append({
                "image_path": img_path,
                "title": row.get("cleaned_title", "Unknown"),
                "similarity": similarities[idx].item(),
                "index": idx
            })
        
        return results
    
    def display_comparison(self, query_image_path: Path):
        """
        Display query image and top-K results from both methods.
        
        Args:
            query_image_path: Path to query image
        """
        # Randomize which method is in row A
        self.baseline_is_row_a = random.random() < 0.5
        
        # Get query embedding using base model (for consistency)
        query_emb_base = self.get_image_embedding(
            query_image_path,
            self.base_model,
            self.base_processor
        )
        
        # Also get from fine-tuned model
        query_emb_finetuned = self.get_image_embedding(
            query_image_path,
            self.finetuned_model,
            self.finetuned_processor
        )
        
        # Perform searches
        base_results = self.search(query_emb_base, method="base", top_k=ArenaConfig.TOP_K)
        finetuned_results = self.search(
            query_emb_finetuned,
            method="finetuned",
            top_k=ArenaConfig.TOP_K
        )
        
        # Assign to rows
        if self.baseline_is_row_a:
            row_a_results = base_results
            row_b_results = finetuned_results
            row_a_label = "Row A: Base CLIP"
            row_b_label = "Row B: Fine-Tuned CLIP"
        else:
            row_a_results = finetuned_results
            row_b_results = base_results
            row_a_label = "Row A: Fine-Tuned CLIP"
            row_b_label = "Row B: Base CLIP"
        
        # Create figure
        if self.fig is not None:
            plt.close(self.fig)
        
        self.fig, axes = plt.subplots(
            2, ArenaConfig.TOP_K + 1,
            figsize=(ArenaConfig.WINDOW_SIZE[0] / 100, ArenaConfig.WINDOW_SIZE[1] / 100),
            dpi=100
        )
        
        # Make it fullscreen
        mngr = plt.get_current_fig_manager()
        try:
            mngr.window.state("zoomed")  # Windows
        except:
            try:
                mngr.full_screen_toggle()  # Linux/Mac
            except:
                pass
        
        # Display query image (spans both rows)
        query_img = Image.open(query_image_path).convert("RGB")
        query_ax = axes[0, 0]
        query_ax.imshow(query_img)
        query_ax.axis("off")
        query_ax.set_title("Query Image", fontsize=14, fontweight="bold")
        
        # Hide second row's first column
        axes[1, 0].axis("off")
        
        # Display Row A results
        for i, result in enumerate(row_a_results, start=1):
            ax = axes[0, i]
            try:
                if result["image_path"].exists():
                    img = Image.open(result["image_path"]).convert("RGB")
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, "Image\nNot Found", 
                           ha="center", va="center", fontsize=10)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{str(e)[:20]}", 
                       ha="center", va="center", fontsize=8)
            
            ax.axis("off")
            title = result["title"][:30] + "..." if len(result["title"]) > 30 else result["title"]
            ax.set_title(f"{title}\n({result['similarity']:.3f})", fontsize=8)
        
        # Display Row B results
        for i, result in enumerate(row_b_results, start=1):
            ax = axes[1, i]
            try:
                if result["image_path"].exists():
                    img = Image.open(result["image_path"]).convert("RGB")
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, "Image\nNot Found", 
                           ha="center", va="center", fontsize=10)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{str(e)[:20]}", 
                       ha="center", va="center", fontsize=8)
            
            ax.axis("off")
            title = result["title"][:30] + "..." if len(result["title"]) > 30 else result["title"]
            ax.set_title(f"{title}\n({result['similarity']:.3f})", fontsize=8)
        
        # Add row labels
        self.fig.text(0.02, 0.75, row_a_label, fontsize=12, fontweight="bold", rotation=90)
        self.fig.text(0.02, 0.25, row_b_label, fontsize=12, fontweight="bold", rotation=90)
        
        # Add instructions
        self.fig.suptitle(
            "Press 'A' for Row A, 'B' for Row B, 'Q' to quit, 'N' for next",
            fontsize=14,
            y=0.98
        )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, left=0.05)
        plt.draw()
        
        self.current_query = query_image_path
    
    def handle_keypress(self, event):
        """
        Handle keyboard input.
        
        Args:
            event: Matplotlib key press event
        """
        if event.key.lower() == "q":
            print("\nQuitting Arena...")
            self.grader.print_statistics()
            plt.close("all")
            return
        
        if event.key.lower() == "n":
            # Next image
            self.next_query()
            return
        
        if event.key.lower() in ["a", "b"]:
            if self.current_query is None:
                print("No query displayed. Press 'N' to load next image.")
                return
            
            # Save vote
            choice = event.key.upper()
            self.grader.save_vote(
                query_image=str(self.current_query),
                baseline_method="Base CLIP",
                challenger_method="Fine-Tuned CLIP",
                user_choice=choice,
                baseline_is_row_a=self.baseline_is_row_a
            )
            
            print(f"Vote recorded: {choice}")
            print("Press 'N' for next image, 'Q' to quit")
            
            # Show statistics
            stats = self.grader.get_statistics()
            print(f"Stats: Baseline {stats['baseline_wins']}/{stats['total_votes']} "
                  f"({stats['baseline_win_rate']*100:.1f}%), "
                  f"Fine-Tuned {stats['challenger_wins']}/{stats['total_votes']} "
                  f"({stats['challenger_win_rate']*100:.1f}%)")
    
    def next_query(self):
        """Load and display next random query image."""
        if len(self.eval_images) == 0:
            print("No more evaluation images available.")
            return
        
        query_image = random.choice(self.eval_images)
        print(f"\nLoading query: {query_image.name}")
        
        try:
            self.display_comparison(query_image)
        except Exception as e:
            print(f"Error displaying query: {e}")
            import traceback
            traceback.print_exception(type(e), e, e.__traceback__)
    
    def run(self):
        """Run the Arena application."""
        print("\n" + "="*60)
        print("Arena Evaluation Interface")
        print("="*60)
        print("Controls:")
        print("  'A' - Vote for Row A")
        print("  'B' - Vote for Row B")
        print("  'N' - Next query image")
        print("  'Q' - Quit and show statistics")
        print("="*60 + "\n")
        
        # Connect keypress handler
        self.fig = plt.figure(figsize=(19.2, 10.8))
        self.fig.canvas.mpl_connect("key_press_event", self.handle_keypress)
        
        # Load first query
        self.next_query()
        
        # Show statistics before starting
        self.grader.print_statistics()
        
        # Start event loop
        plt.show()


def main():
    """Main entry point for Arena application."""
    app = ArenaApp()
    app.run()


if __name__ == "__main__":
    main()

