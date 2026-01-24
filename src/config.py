"""
Central configuration file for the Furniture Semantic Search Engine.

This module contains all paths, hyperparameters, and constants used throughout
the project. Update these values to match your local setup.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent


class DataPaths:
    """Paths for data directories and files."""
    
    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    RAW_CSV_PATH = RAW_DIR / "furniture_dataset_cleaned.csv"
    IMAGE_DIR = RAW_DIR / "downloaded_images"
    
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    CLEANED_CSV_PATH = PROCESSED_DIR / "cleaned_furniture.csv"
    
    EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
    MASTER_VECTORS_PATH = EMBEDDINGS_DIR / "master_vectors.parquet"
    
    EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
    EVALUATION_IMAGE_DIR = EVALUATION_DIR
    
    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories if they don't exist."""
        cls.RAW_DIR.mkdir(parents=True, exist_ok=True)
        (cls.RAW_DIR / "downloaded_images").mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cls.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


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


class ModelConfig:
    """CLIP model configuration."""
    
    MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
    IMAGE_SIZE = 224
    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]
    MAX_TEXT_LENGTH = 77


class TrainingConfig:
    """Hyperparameters for CLIP fine-tuning."""
    
    BATCH_SIZE = 32
    VAL_BATCH_SIZE = 64
    EMBEDDING_BATCH_SIZE = 128
    
    INITIAL_LR = 1e-4
    UNFROZEN_LR = 5e-6
    
    EPOCHS = 5
    FREEZE_EPOCHS = 1
    
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 500
    
    VAL_SPLIT = 0.1
    
    DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    
    SAVE_EVERY_N_EPOCHS = 1
    EVAL_EVERY_N_STEPS = 500


class CleaningConfig:
    """Configuration for GPT-5 nano VLM batch cleaning."""
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = "gpt-5-nano"
    
    BATCH_INPUT_PATH = PROJECT_ROOT / "data" / "batch_workspace" / "input_jsonl"
    BATCH_OUTPUT_DIR = PROJECT_ROOT / "data" / "batch_workspace" / "output_jsonl"
    CLEANED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_furniture.csv"
    RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "furniture_dataset_cleaned.csv"
    IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "downloaded_images"
    BATCH_WORKSPACE = PROJECT_ROOT / "data" / "batch_workspace"
    BATCH_INPUT_DIR = BATCH_WORKSPACE / "input_jsonl"
    STATE_FILE = BATCH_WORKSPACE / "batch_state.json"
    
    MAX_BATCH_FILE_SIZE = 90 * 1024 * 1024
    MAX_REQUESTS_PER_FILE = 49000
    MAX_CONCURRENT_BATCHES = 1
    POLL_INTERVAL = 600
    
    IMAGE_RESIZE_DIM = 512

    @classmethod
    def ensure_dirs(cls):
        cls.BATCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CLEANED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)


class PrepConfig:
    """Data formatting configuration."""
    
    INPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_furniture.csv"
    IMAGES_ROOT_DIR = PROJECT_ROOT / "data" / "raw"
    OUTPUT_DIR = PROJECT_ROOT / "data" / "webdataset_shards"

    IMAGE_SIZE = 224
    PADDING_COLOR = (255, 255, 255)
    IMAGE_QUALITY = 95

    SEED = 42
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

    MAX_SHARD_SIZE = 500 * 1024 * 1024
    MAX_COUNT_PER_SHARD = 1000

    MINI_TRAIN_SHARD_COUNT = 5


class EmbeddingConfig:
    """Configuration for embedding generation."""
    
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    DEVICE = TrainingConfig.DEVICE


class ArenaConfig:
    """Configuration for the evaluation Arena."""
    
    RESULTS_CSV = PROJECT_ROOT / "arena_results.csv"
    TOP_K = 4
    WINDOW_SIZE = (1920, 1080)
    RANDOMIZE_ORDER = True


def initialize_directories():
    """Create all necessary directories."""
    DataPaths.ensure_dirs()
    ModelPaths.ensure_dirs()


initialize_directories()

