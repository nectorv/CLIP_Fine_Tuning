"""
Central configuration file for the Furniture Semantic Search Engine.

This module contains all paths, hyperparameters, and constants used throughout
the project. Update these values to match your local setup.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PROJECT ROOT
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# DATA PATHS
# ============================================================================
class DataPaths:
    """Paths for data directories and files."""
    
    # Raw data directories (within project)
    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    RAW_CSV_PATH = RAW_DIR / "furniture_dataset_cleaned.csv"
    IMAGE_DIR = RAW_DIR / "downloaded_images"  # Training images
    
    # Processed data
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    CLEANED_CSV_PATH = PROCESSED_DIR / "cleaned_furniture.csv"
    
    # Embeddings
    EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
    MASTER_VECTORS_PATH = EMBEDDINGS_DIR / "master_vectors.parquet"
    
    # Evaluation
    EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
    EVALUATION_IMAGE_DIR = EVALUATION_DIR  # Evaluation images
    
    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories if they don't exist."""
        cls.RAW_DIR.mkdir(parents=True, exist_ok=True)
        (cls.RAW_DIR / "downloaded_images").mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cls.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MODEL PATHS
# ============================================================================
class ModelPaths:
    """Paths for model storage."""
    
    MODELS_DIR = PROJECT_ROOT / "models"
    FINETUNED_DIR = MODELS_DIR / "clip-finetuned"
    CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
    
    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories if they don't exist."""
        cls.FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
class ModelConfig:
    """CLIP model configuration."""
    
    # Base model from HuggingFace
    MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
    
    # Image preprocessing
    IMAGE_SIZE = 224
    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]
    
    # Text preprocessing
    MAX_TEXT_LENGTH = 77  # CLIP's default


# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
class TrainingConfig:
    """Hyperparameters for CLIP fine-tuning."""
    
    # Batch sizes
    BATCH_SIZE = 32
    VAL_BATCH_SIZE = 64
    EMBEDDING_BATCH_SIZE = 128  # For faster embedding generation
    
    # Learning rates
    INITIAL_LR = 1e-4  # For projection layers (frozen backbone)
    UNFROZEN_LR = 5e-6  # For unfrozen model (lower LR)
    
    # Training schedule
    EPOCHS = 5
    FREEZE_EPOCHS = 1  # Number of epochs with frozen backbones
    
    # Optimization
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 500
    
    # Validation
    VAL_SPLIT = 0.1  # 10% for validation
    
    # Device
    DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Checkpointing
    SAVE_EVERY_N_EPOCHS = 1
    EVAL_EVERY_N_STEPS = 500


# ============================================================================
# DATA CLEANING CONFIGURATION
# ============================================================================
class CleaningConfig:
    """Configuration for GPT-5 nano VLM batch cleaning."""
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = "gpt-5-nano"  # Vision Language Model
    
    # Batch API paths
    BATCH_INPUT_PATH = PROJECT_ROOT / "data" / "batch_workspace" / "input_jsonl"
    BATCH_OUTPUT_DIR = PROJECT_ROOT / "data" / "batch_workspace" / "output_jsonl"
    CLEANED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_furniture.csv"
    RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "furniture_dataset_cleaned.csv"
    IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "downloaded_images"  # Training images
    BATCH_WORKSPACE = PROJECT_ROOT / "data" / "batch_workspace"
    BATCH_INPUT_DIR = BATCH_WORKSPACE / "input_jsonl"
    STATE_FILE = BATCH_WORKSPACE / "batch_state.json"
    
    # Limits
    MAX_BATCH_FILE_SIZE = 90 * 1024 * 1024  # 90MB
    MAX_REQUESTS_PER_FILE = 49000
    MAX_CONCURRENT_BATCHES = 1  # To manage queue/token limits
    POLL_INTERVAL = 300  # Seconds
    
    # Image Processing
    IMAGE_RESIZE_DIM = 512


    @classmethod
    def ensure_dirs(cls):
        cls.BATCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CLEANED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================
class EmbeddingConfig:
    """Configuration for embedding generation."""
    
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    DEVICE = TrainingConfig.DEVICE


# ============================================================================
# ARENA CONFIGURATION
# ============================================================================
class ArenaConfig:
    """Configuration for the evaluation Arena."""
    
    RESULTS_CSV = PROJECT_ROOT / "arena_results.csv"
    TOP_K = 4  # Number of results to display per method
    WINDOW_SIZE = (1920, 1080)  # Full-screen window
    RANDOMIZE_ORDER = True  # Blind test mode


# ============================================================================
# INITIALIZATION
# ============================================================================
def initialize_directories():
    """Create all necessary directories."""
    DataPaths.ensure_dirs()
    ModelPaths.ensure_dirs()


# Initialize on import
initialize_directories()

