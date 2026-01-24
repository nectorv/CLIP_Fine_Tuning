# API Documentation

## Core Modules

### src.config

Centralized configuration management for the entire project.

#### Classes

##### `DataPaths`
Manages all data directory paths.

```python
from src.config import DataPaths

# Access paths
raw_csv = DataPaths.RAW_CSV_PATH
image_dir = DataPaths.IMAGE_DIR
embeddings = DataPaths.EMBEDDINGS_DIR

# Ensure directories exist
DataPaths.ensure_dirs()
```

**Attributes:**
- `RAW_DIR`: Raw data directory
- `RAW_CSV_PATH`: Path to raw CSV file
- `IMAGE_DIR`: Directory containing images
- `PROCESSED_DIR`: Processed data directory
- `CLEANED_CSV_PATH`: Path to cleaned CSV
- `EMBEDDINGS_DIR`: Embeddings storage directory
- `EVALUATION_DIR`: Evaluation images directory

##### `ModelConfig`
CLIP model configuration and preprocessing parameters.

```python
from src.config import ModelConfig

model_name = ModelConfig.MODEL_NAME
image_size = ModelConfig.IMAGE_SIZE
```

**Attributes:**
- `MODEL_NAME`: HuggingFace model identifier
- `IMAGE_SIZE`: Image dimension for preprocessing
- `IMAGE_MEAN`: Normalization mean values
- `IMAGE_STD`: Normalization standard deviation
- `MAX_TEXT_LENGTH`: Maximum text token length

##### `TrainingConfig`
Training hyperparameters and optimization settings.

```python
from src.config import TrainingConfig

lr = TrainingConfig.INITIAL_LR
batch_size = TrainingConfig.BATCH_SIZE
epochs = TrainingConfig.EPOCHS
```

**Attributes:**
- `BATCH_SIZE`: Training batch size
- `VAL_BATCH_SIZE`: Validation batch size
- `EMBEDDING_BATCH_SIZE`: Batch size for embedding generation
- `INITIAL_LR`: Initial learning rate
- `EPOCHS`: Number of training epochs
- `WEIGHT_DECAY`: L2 regularization coefficient
- `WARMUP_STEPS`: Warmup phase length
- `DEVICE`: Computing device (cuda/cpu)

### src.training.main

Model training orchestration with distributed support.

#### Functions

##### `parse_args()`
Parse command-line arguments for training configuration.

```python
args = parse_args()
print(args.scenario)  # Training scenario
print(args.lr)        # Learning rate
print(args.epochs)    # Number of epochs
```

**Returns:** `argparse.Namespace` with configuration

##### `main()`
Execute the complete training pipeline.

```python
main()
# Initializes WandB, downloads data, trains model, saves checkpoints
```

### src.cleaning.api_client

OpenAI Batch API client wrapper.

#### Class

##### `BatchAPIClient`
Manages communication with OpenAI Batch API.

```python
from src.cleaning.api_client import BatchAPIClient

client = BatchAPIClient()

# Upload file
file_id = client.upload_file(Path("batch_input.jsonl"))

# Create batch
batch_id = client.create_batch(file_id)

# Check status
status = client.get_batch_status(batch_id)

# Download results
results = client.download_results(output_file_id)
```

**Methods:**
- `upload_file(file_path: Path) -> str`: Upload file and return file ID
- `create_batch(input_file_id: str) -> str`: Create batch job
- `get_batch_status(batch_id: str) -> dict`: Get batch status
- `download_results(output_file_id: str) -> bytes`: Download results

### src.cleaning.cleaner

Batch data cleaning using GPT-5 nano VLM.

#### Classes

##### `CleanedProduct`
Pydantic schema for cleaned product data.

```python
from src.cleaning.cleaner import CleanedProduct

product = CleanedProduct(
    clean_title="Modern Sofa",
    style="Contemporary",
    material="Leather",
    color="Black",
    object_type="Sofa"
)
```

#### Functions

##### `prepare_batch_jsonl()`
Prepare batch input files for OpenAI Batch API.

```python
from src.cleaning.cleaner import prepare_batch_jsonl
from pathlib import Path

count = prepare_batch_jsonl(
    csv_path=Path("data/raw/furniture.csv"),
    image_dir=Path("data/raw/images"),
    output_dir=Path("batch_input"),
    incremental=True
)
print(f"Created {count} requests")
```

**Parameters:**
- `csv_path`: Input CSV file path
- `image_dir`: Directory containing product images
- `output_dir`: Output directory for JSONL files
- `title_column`: CSV column name for titles
- `image_column`: CSV column name for image paths
- `incremental`: Skip already-processed items

**Returns:** Number of requests created

##### `upload_batch()`
Upload batch file to OpenAI.

```python
from src.cleaning.cleaner import upload_batch

batch_id = upload_batch(Path("batch_input_001.jsonl"))
print(f"Batch uploaded: {batch_id}")
```

##### `check_status()`
Check batch processing status.

```python
from src.cleaning.cleaner import check_status

status = check_status("batch_123456")
print(status["status"])  # processing, completed, failed
```

##### `finalize_results()`
Process batch results and merge with original CSV.

```python
from src.cleaning.cleaner import finalize_results

df = finalize_results(
    batch_output_path=Path("batch_output_001.jsonl"),
    original_csv_path=Path("data/raw/furniture.csv"),
    output_csv_path=Path("data/processed/cleaned.csv")
)
```

### src.preparing.sharding

WebDataset sharding for efficient data loading.

#### Functions

##### `write_dataset()`
Create WebDataset shards from DataFrame.

```python
from src.preparing.sharding import write_dataset
from src.config import PrepConfig

write_dataset(
    df=train_df,
    output_folder="data/shards/train",
    image_root_dir="data/raw",
    config=PrepConfig
)
```

**Parameters:**
- `df`: Input DataFrame with samples
- `output_folder`: Output directory for shards
- `image_root_dir`: Root directory for relative image paths
- `config`: Configuration object with shard settings

##### `create_mini_train()`
Create mini-training dataset from full training set.

```python
from src.preparing.sharding import create_mini_train

create_mini_train(
    train_folder="data/shards/train",
    mini_folder="data/shards/mini_train",
    num_shards=5
)
```

### src.embeddings.embedder

Embedding generation for images and text.

#### Class

##### `EmbeddingGenerator`
Generate embeddings using base and fine-tuned models.

```python
from src.embeddings.embedder import EmbeddingGenerator

generator = EmbeddingGenerator()

# Generate image embeddings
image_embeddings = generator.generate_image_embeddings(
    model=model,
    processor=processor,
    image_paths=image_list,
    embedding_save_path="embeddings.parquet"
)

# Generate text embeddings
text_embeddings = generator.generate_text_embeddings(
    model=model,
    processor=processor,
    texts=text_list
)
```

**Methods:**
- `generate_image_embeddings()`: Generate embeddings from images
- `generate_text_embeddings()`: Generate embeddings from text
- `generate_all_embeddings()`: Batch process entire dataset

### src.arena.app

Interactive evaluation GUI.

#### Class

##### `ArenaApp`
Full-screen comparison interface for model evaluation.

```python
from src.arena.app import ArenaApp

app = ArenaApp()
app.next_query()
# User compares baseline vs fine-tuned model results
# Press Q to quit and view statistics
```

**Methods:**
- `next_query()`: Load next random query image
- `display_comparison()`: Display side-by-side results
- `handle_keypress()`: Process keyboard input
- `display_results()`: Show evaluation statistics

**Controls:**
- **N**: Next image
- **A/B**: Vote for preference
- **Q**: Quit and show results

### src.arena.grading

Evaluation result tracking and statistics.

#### Class

##### `ArenaGrader`
Manage evaluation votes and statistics.

```python
from src.arena.grading import ArenaGrader

grader = ArenaGrader()

# Save vote
grader.save_vote(
    query_image="sofa_001.jpg",
    baseline_method="Base CLIP",
    challenger_method="Fine-Tuned CLIP",
    user_choice="B",
    baseline_is_row_a=True
)

# Get statistics
stats = grader.compute_statistics()
```

**Methods:**
- `save_vote()`: Record user vote
- `compute_statistics()`: Calculate win rates
- `print_statistics()`: Display results summary

## Data Flow

### Cleaning Pipeline
```
Raw CSV + Images
        ↓
prepare_batch_jsonl()
        ↓
Batch API (OpenAI)
        ↓
finalize_results()
        ↓
Cleaned CSV
```

### Training Pipeline
```
Cleaned CSV + Images
        ↓
write_dataset()
        ↓
WebDataset Shards
        ↓
Training Loop
        ↓
Fine-tuned Model
```

### Evaluation Pipeline
```
Fine-tuned Model + Test Images
        ↓
EmbeddingGenerator
        ↓
Master Vectors
        ↓
Arena Interface
        ↓
Evaluation Results
```

## Example Workflows

### Complete Data Cleaning
```python
from pathlib import Path
from src.cleaning.cleaner import (
    prepare_batch_jsonl,
    upload_batch,
    check_status,
    wait_for_completion,
    finalize_results
)

# Prepare
count = prepare_batch_jsonl(
    Path("data/raw/furniture.csv"),
    Path("data/raw/images"),
    Path("batch_input")
)
print(f"Prepared {count} requests")

# Upload
batch_id = upload_batch(Path("batch_input_001.jsonl"))
print(f"Batch ID: {batch_id}")

# Monitor
if wait_for_completion(batch_id, max_wait_hours=24):
    # Finalize
    df = finalize_results(
        Path("batch_output_001.jsonl"),
        Path("data/raw/furniture.csv"),
        Path("data/processed/cleaned.csv")
    )
    print(f"Cleaned {len(df)} records")
```

### Training with Custom Config
```python
import torch
from src.training.model import get_model
from src.training.data import get_wds_pipeline

# Load model with LoRA
model, processor = get_model("dual_lora", lora_r=16)
model.to("cuda")

# Prepare data
dataset = get_wds_pipeline("data/shards/train", processor, is_train=True)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for epoch in range(5):
    for batch in dataset:
        outputs = model(**batch, return_loss=True)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## Performance Tips

1. **Batch Size**: Increase for better GPU utilization
2. **Gradient Accumulation**: For effective larger batches on limited memory
3. **Mixed Precision**: Use AMP for ~2x speedup
4. **Distributed Training**: Use DataParallel for multi-GPU
5. **Data Loading**: Use WebDataset for efficient streaming

## Error Handling

All modules include comprehensive error handling:

```python
try:
    cleaner = prepare_batch_jsonl(...)
except FileNotFoundError:
    print("CSV file not found")
except Exception as e:
    print(f"Error: {e}")
```

## Thread Safety

- `BatchAPIClient`: Thread-safe for concurrent API calls
- `EmbeddingGenerator`: Single-threaded (use multiprocessing)
- `ArenaGrader`: Thread-safe file operations
