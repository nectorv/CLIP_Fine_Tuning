from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class CleaningConfig:
    # API
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = "gpt-5-nano"  # Or your specific model version
    
    # Limits
    MAX_BATCH_FILE_SIZE = 90 * 1024 * 1024  # 90MB
    MAX_REQUESTS_PER_FILE = 49000
    MAX_CONCURRENT_BATCHES = 5  # To manage queue/token limits
    POLL_INTERVAL = 60  # Seconds
    
    # Image Processing
    IMAGE_RESIZE_DIM = 512
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    
    RAW_CSV_PATH = DATA_DIR / "raw" / "furniture_dataset_cleaned.csv"
    IMAGE_DIR = DATA_DIR / "images"
    
    # Output Areas
    BATCH_WORKSPACE = DATA_DIR / "batch_workspace"
    BATCH_INPUT_DIR = BATCH_WORKSPACE / "input_jsonl"
    BATCH_OUTPUT_DIR = BATCH_WORKSPACE / "output_jsonl"
    STATE_FILE = BATCH_WORKSPACE / "batch_state.json"
    CLEANED_CSV_PATH = DATA_DIR / "cleaned" / "cleaned_products.csv"

    @classmethod
    def ensure_dirs(cls):
        cls.BATCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CLEANED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)