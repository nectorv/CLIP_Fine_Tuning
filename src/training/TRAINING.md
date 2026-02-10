# CLIP Fine-Tuning Training System

## Overview

The training system is designed to fine-tune OpenAI's CLIP model (Vision-Language model) on a furniture dataset using various training scenarios. The implementation leverages advanced techniques including LoRA (Low-Rank Adaptation), mixed precision training, gradient accumulation, and WebDataset for efficient large-scale data handling.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components](#key-components)
3. [Training Scenarios](#training-scenarios)
4. [Data Pipeline](#data-pipeline)
5. [Model Configuration](#model-configuration)
6. [Training Loop](#training-loop)
7. [Optimization Techniques](#optimization-techniques)
8. [Infrastructure & S3 Integration](#infrastructure--s3-integration)
9. [Configuration Parameters](#configuration-parameters)
10. [Workflow](#workflow)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIP Fine-Tuning Pipeline                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Data Source (S3) → Download → WebDataset Pipeline           │
│                                      ↓                        │
│                              Augmentation & Processing        │
│                                      ↓                        │
│                              CLIP Processor (Tokenization)   │
│                                      ↓                        │
│  Model (LoRA/Linear Probe) ← Batching & DataLoader           │
│                                      ↓                        │
│  Training Loop (Mixed Precision, Gradient Accumulation)      │
│                                      ↓                        │
│  Validation & Monitoring (WandB)                             │
│                                      ↓                        │
│  Checkpoint Management (Local + Optional S3)                 │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. **main.py** - Training Orchestration
The entry point that coordinates the entire training pipeline:
- Argument parsing and configuration
- S3 data synchronization
- Model and data loading
- Training loop execution
- Checkpoint management

### 2. **model.py** - Model Architecture
Defines different fine-tuning scenarios by selectively unfreezing parameters:
- Base model loading (OpenAI CLIP ViT-base-patch32)
- Scenario-specific parameter freezing strategies
- LoRA configuration for efficient adaptation

### 3. **data.py** - Data Pipeline
Implements WebDataset-based data loading:
- Shard discovery and loading
- Image-text preprocessing
- Data augmentation strategies
- Integration with CLIP processor

### 4. **utils.py** - Utility Functions
Provides support utilities:
- **S3Manager**: Download datasets and optionally upload/resume checkpoints
- **EarlyStopper**: Early stopping logic
- **find_max_batch_size()**: Automatic batch size discovery

---

## Training Scenarios

The system supports multiple training scenarios for ablation studies:

### 1. **Zero-Shot**
- **Description**: No fine-tuning, frozen base model
- **Frozen Parameters**: All parameters frozen
- **Use Case**: Baseline performance comparison
- **Training Time**: N/A (inference only)

### 2. **Linear Probe**
- **Description**: Train only projection layers
- **Unfrozen Parameters**: 
  - `model.visual_projection`
  - `model.text_projection`
- **Use Case**: Quick adaptation with minimal parameter changes
- **Trainable Parameters**: ~2-3% of total

### 3. **Dual LoRA** (Default)
- **Description**: LoRA on both vision and text encoders + unfrozen projections
- **Configuration**:
  - LoRA rank (r): 16 (default)
  - LoRA alpha: 32 (default)
  - Target modules: `q_proj`, `v_proj` (attention heads)
  - Dropout: 0.05 (default)
- **Unfrozen Parameters**: 
  - Both vision and text projections
  - LoRA parameters in attention mechanisms
- **Trainable Parameters**: ~3-5% of total
- **Advantage**: Efficient parameter count with strong performance

### 4. **Vision LoRA**
- **Description**: LoRA on vision encoder only
- **Trainable Parameters**: ~2% of total
- **Use Case**: When text embeddings are well-aligned

### 5. **Text LoRA**
- **Description**: LoRA on text encoder only
- **Trainable Parameters**: ~2% of total
- **Use Case**: When vision embeddings are well-aligned

---

## Data Pipeline

### WebDataset Integration

The data pipeline uses **WebDataset** (wds) for efficient handling of large-scale image-text pairs:

```
Raw Data (S3)
    ↓
Shards (.tar files)
    ↓
WebDataset Pipeline
    ├── Read shards
    ├── Decode PIL images
    ├── Extract image + text pairs
    ├── Apply augmentation
    ├── CLIP Processor (tokenization + normalization)
    └── Return batches
```

### Data Format

Each shard contains:
- **Images**: `.jpg` files (raw bytes)
- **Text**: `.txt` files (prompt descriptions)
- **Metadata**: `.json` files (ignored during training)

### Augmentation Strategies

#### Simple Augmentation
```python
- Resize to 224×224
- Random horizontal flip
- Random crop with padding
```

#### Advanced Augmentation
```python
- Resize to 224×224
- TrivialAugmentWide (automatic augmentation)
- Random erasing (20% probability)
- Supports tensor/PIL conversion
```

### CLIP Processor

The CLIP processor handles:
1. **Image Processing**:
   - Normalization (ImageNet mean/std)
   - Conversion to tensors
   - Output: `pixel_values` tensor

2. **Text Processing**:
   - Tokenization
   - Padding to max_length
   - Truncation if needed
   - Output: `input_ids` tensor

---

## Model Configuration

### Base Model
- **Architecture**: CLIP Vision Transformer (ViT) + Text Transformer
- **Model ID**: `openai/clip-vit-base-patch32`
- **Parameters**: ~349M total
- **Vision Encoder**: ViT-Base with 32×32 patches
- **Text Encoder**: Transformer-based
- **Projection Layers**: 512-dimensional embeddings

### LoRA Configuration

**LoRA (Low-Rank Adaptation)** decomposes weight matrices into low-rank updates:

```
W = W₀ + ΔW = W₀ + BA

where:
  B: Down-projection (d_model × r)
  A: Up-projection (r × d_model)
  r: Rank (typically 8-32)
```

**Benefits**:
- ✅ Reduces trainable parameters from millions to thousands
- ✅ Lower memory footprint during training
- ✅ Faster training and inference
- ✅ Maintains model expressivity

**Default Configuration**:
```
r (rank):           16      # Dimension of low-rank matrices
alpha:              32      # Scaling factor (2 × r typical)
dropout:            0.05    # Regularization
target_modules:     ["q_proj", "v_proj"]  # Attention query/value projections
bias:               "none"  # Don't add LoRA to bias
```

---

## Training Loop

### Phases

#### 1. **Initialization Phase**
```python
# Load data from S3
s3_mgr.download_dataset(s3_train_prefix, local_train)
s3_mgr.download_dataset(s3_val_prefix, local_val)

# Load model with selected scenario
model, processor = get_model(scenario, lora_r)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=micro_bs)
val_loader = DataLoader(val_dataset, batch_size=micro_bs)
```

#### 2. **Training Loop (Per Epoch)**
```
for epoch in range(num_epochs):
    # Training phase
    for step, batch in enumerate(train_loader):
        ├── Extract pixel_values and input_ids
        ├── Mixed precision forward pass
        ├── Compute contrastive loss
        ├── Scale loss for gradient accumulation
        ├── Backward pass
        ├── Gradient clipping
        ├── Optimizer step (when accumulation reaches target)
        └── Learning rate scheduler step
    
    # Validation phase
    for batch in val_loader:
        ├── Forward pass without gradients
        ├── Compute validation loss
        └── Accumulate metrics
    
    ├── Log metrics to WandB
    ├── Save checkpoint locally & to S3
    └── Check early stopping condition
```

#### 3. **Key Operations**

##### Loss Computation
```python
outputs = model(input_ids=input_ids, pixel_values=pixel_values, return_loss=True)
loss = outputs.loss / grad_accum_steps  # Scale for accumulation
```

The CLIP loss is a symmetric cross-entropy loss between:
- Image embeddings → Text embeddings
- Text embeddings → Image embeddings

##### Gradient Accumulation
```
Effective Batch Size = Micro Batch Size × Gradient Accumulation Steps
Example: 64 × 16 = 1024

This allows training with large effective batches on limited GPU memory.
```

##### Gradient Clipping
```python
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Prevents exploding gradients by clipping to max norm of 1.0.

---

## Optimization Techniques

### 1. **Mixed Precision Training (AMP)**
Uses automatic mixed precision to combine FP32 and FP16:
```python
with autocast():  # FP16 computations
    outputs = model(...)
    loss = outputs.loss

scaler.scale(loss).backward()  # FP32 backward pass
```

**Benefits**:
- 40-50% faster training
- Reduced memory usage
- Improved numerical stability with loss scaling

### 2. **8-bit Optimization (BitsAndBytes)**
```python
optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.wd
)
```

**Benefits**:
- Reduces optimizer memory by 75%
- Maintains full precision weight updates
- Ideal for large models on limited memory

### 3. **Learning Rate Scheduling**
Uses OneCycleLR scheduler:
```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=args.lr,
    total_steps=total_steps,
    pct_start=0.1  # 10% warmup
)
```

**Profile**:
1. Linear warmup (10% of training)
2. Gradual decay to min_lr
3. Final increase to max_lr

### 4. **Gradient Accumulation**
Simulates larger batch sizes by accumulating gradients over multiple steps:
```
Accumulation Steps = Effective Batch Size / Micro Batch Size
```

---

## Infrastructure & S3 Integration

### S3 Manager

The `S3Manager` class handles cloud storage operations:

#### Download Dataset
```python
s3_mgr.download_dataset(s3_prefix, local_dir)
# Uses: aws s3 sync (faster than boto3 loop for 50k+ files)
```

#### Optional Checkpoints (S3)
```python
if args.enable_s3_checkpoints:
    s3_mgr.upload_checkpoint(local_path, s3_path)
```

### Checkpoint Format
```python
checkpoint = {
    'epoch': current_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': validation_loss
}
```

---

## Configuration Parameters

### Data & Infrastructure
| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_dir` | `/tmp/furniture_data` | Local path for dataset |
| `s3_bucket` | *required* | S3 bucket name |
| `s3_prefix` | `data/train` | S3 folder path |
| `output_dir` | `./checkpoints` | Local checkpoint directory |
| `mini_train` | False | Use subset for testing |

### Training Setup
| Parameter | Default | Description |
|-----------|---------|-------------|
| `scenario` | `dual_lora` | Training scenario |
| `epochs` | 10 | Number of epochs |
| `effective_batch_size` | 1024 | Target accumulated batch size |
| `micro_batch_size` | 0 (auto) | Physical GPU batch size |
| `num_workers` | 4 | DataLoader workers |
| `seed` | 42 | Random seed |

### Optimization
| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-4 | Learning rate |
| `wd` | 0.1 | Weight decay |
| `warmup_steps` | 100 | Linear warmup steps |
| `grad_clip` | 1.0 | Gradient clipping norm |
| `augmentation` | `simple` | Augmentation strategy |

### LoRA Specifics
| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA scaling factor |
| `lora_dropout` | 0.05 | LoRA dropout |

### Monitoring
| Parameter | Default | Description |
|-----------|---------|-------------|
| `wandb_project` | `clip-furniture-finetune` | W&B project name |
| `enable_s3_checkpoints` | False | Enable upload/resume via S3 checkpoints |
| `resume_from_checkpoint` | False | Resume from S3 checkpoint (requires `enable_s3_checkpoints`) |

---

## Workflow

### Step-by-Step Training Process

#### 1. **Preparation**
```bash
python main.py \
    --s3_bucket my-bucket \
    --scenario dual_lora \
    --epochs 10 \
    --lr 1e-4
```

#### 2. **Data Download**
- Syncs training set from S3 (45,000 samples)
- Syncs validation set from S3 (5,000 samples)
- Creates sharded WebDataset format

#### 3. **Model Loading**
- Loads CLIP ViT-base-patch32 from HuggingFace
- Applies scenario-specific freezing strategy
- For LoRA scenarios: Wraps model with PEFT

#### 4. **Batch Size Discovery** (if micro_batch_size=0)
- Tests increasing batch sizes until OOM
- Returns maximum sustainable batch size
- Fallback: 64 for WebDataset

#### 5. **Training Loop** (per epoch)
- Samples from training dataset with shuffling
- Accumulates gradients for effective_batch_size
- Applies mixed precision and 8-bit optimization
- Every accumulation step:
  - Unscales and clips gradients
  - Updates weights
  - Steps learning rate scheduler

#### 6. **Validation** (end of epoch)
- Evaluates on full validation set
- Computes average loss
- Logs to WandB

#### 7. **Checkpoint Management**
- Saves locally: `{scenario}_checkpoint.pt`
- Optionally uploads to S3 when `enable_s3_checkpoints` is set
- Tracks: epoch, model weights, optimizer state, loss

#### 8. **Early Stopping**
- Monitors validation loss
- Stops if no improvement for 2 epochs
- Threshold: 0.001 minimum improvement

---

## Performance Considerations

### Memory Optimization
- **LoRA**: Reduces trainable params from 349M to ~10-15M
- **8-bit Optimizer**: 75% reduction in optimizer memory
- **Mixed Precision**: ~50% memory reduction
- **Gradient Accumulation**: Allows larger effective batches

### Speed Optimization
- **WebDataset**: Efficient I/O for large-scale data
- **Mixed Precision**: 40-50% training speedup
- **Batch Finder**: Auto-discovers optimal batch size
- **S3 Sync**: Faster than boto3 for large datasets

### Convergence
- **OneCycleLR**: Improves final performance and convergence
- **Warmup**: Stabilizes initial training
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Prevents overfitting

---

## Monitoring & Logging

### WandB Integration
Logs key metrics for experiment tracking:
- **Training Loss**: Per gradient accumulation step
- **Validation Loss**: End of epoch
- **Learning Rate**: Scheduler progression
- **Epoch**: Current training epoch

### Metrics Tracked
```python
wandb.log({
    "val_loss": avg_val_loss,
    "epoch": epoch,
    "lr": current_learning_rate  # optional
})
```

### Resumption Information
```
⏩ Resuming from epoch {start_epoch}
```
Only displayed when resuming from checkpoint.

---

## Error Handling

### Common Issues & Solutions

#### Out of Memory (OOM)
1. Reduce `micro_batch_size`
2. Reduce `effective_batch_size`
3. Enable gradient accumulation
4. Use mixed precision (default enabled)

#### No Data Found
```
❌ Aucun fichier .tar trouvé dans {data_dir}
```
Ensure S3 sync completed successfully or data is locally available.

#### Checkpoint Issues
```
⚠️ No checkpoint found in S3 to resume from.
```
Training will start from epoch 0 if no checkpoint exists.

#### Learning Rate Issues
- Start with 1e-4 for LoRA fine-tuning
- Use 1e-3 for linear probe (wider updates)
- Monitor with WandB graphs

---

## Architecture Diagram: Training Loop

```
┌────────────────────────────────────────────────────────────┐
│                    Epoch Loop                              │
└────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴────────────┐
                ▼                        ▼
        ┌─────────────────┐      ┌─────────────────┐
        │ Training Phase  │      │ Validation      │
        └─────────────────┘      │ Phase           │
                │                └─────────────────┘
                │                         │
        ┌───────┴────────────┐           │
        ▼                    ▼           ▼
   ┌──────────┐      ┌─────────────┐ ┌──────────┐
   │ Forward  │      │ Accumulate  │ │ Compute  │
   │ Pass     │      │ Gradients   │ │ Val Loss │
   └────┬─────┘      └─────────────┘ └──────────┘
        │                │                │
        ▼                ▼                │
   ┌──────────┐      ┌─────────────┐     │
   │ Mixed    │      │ Clip &      │     │
   │ Precision│      │ Optimize    │     │
   └────┬─────┘      └─────────────┘     │
        │                │                │
        ▼                ▼                │
   ┌──────────┐      ┌─────────────┐     │
   │ Backward │      │ Schedule LR │     │
   │ Pass     │      └─────────────┘     │
   └──────────┘                          │
                                         │
                ┌────────────────────────┴───┐
                ▼                            ▼
        ┌─────────────────┐        ┌──────────────────┐
        │ Save            │        │ Check Early      │
        │ Checkpoint      │        │ Stopping         │
        │ (Local + S3)    │        └──────────────────┘
        └─────────────────┘
```

---

## Conclusion

The CLIP fine-tuning system provides a production-ready framework for adapting CLIP to specific domains (e.g., furniture recognition) with:

✅ **Efficiency**: LoRA + mixed precision + 8-bit optimization  
✅ **Scalability**: WebDataset for handling 50k+ image-text pairs  
✅ **Flexibility**: Multiple training scenarios for ablation studies  
✅ **Reliability**: Checkpoint management and early stopping  
✅ **Observability**: WandB integration for experiment tracking  

This architecture balances performance, memory efficiency, and training speed for practical deployment.
