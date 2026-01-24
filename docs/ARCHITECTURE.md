# Architecture Overview

## System Design

CLIP Fine-Tuning follows a modular pipeline architecture with four independent stages:

```
┌─────────────┐       ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│   Cleaning  │──────▶│ Preparation  │──────▶│   Training   │──────▶│  Evaluation  │
├─────────────┤       ├──────────────┤       ├──────────────┤       ├──────────────┤
│   GPT-5     │       │  WebDataset  │       │   Fine-tune  │       │    Arena     │
│   Batch API │       │   Sharding   │       │     LoRA     │       │    GUI       │
└─────────────┘       └──────────────┘       └──────────────┘       └──────────────┘
```

## Component Architecture

### 1. Data Cleaning Pipeline

**Purpose**: Extract structured metadata from raw furniture data

**Flow**:
```
Raw CSV + Images
    │
    ├─ Resize & encode images to base64
    ├─ Create JSONL with vision prompts
    ├─ Split into <95MB chunks (OpenAI limit)
    │
    ▼
Upload to OpenAI Batch API
    │
    ├─ Status: in_progress
    ├─ Polling: Every 10 minutes
    │
    ▼
Download & Parse Results
    │
    ├─ Extract structured fields
    ├─ Merge with original CSV
    │
    ▼
Cleaned CSV
  ├─ clean_title
  ├─ style
  ├─ material
  ├─ color
  └─ object_type
```

**Key Classes**:
- `BatchAPIClient`: OpenAI Batch API wrapper
- `StateManager`: Track batch processing state
- `DataProcessor`: JSON/CSV operations

**Configuration** ([src/config.py](../src/config.py#L85-L110)):
```python
CleaningConfig:
  - OPENAI_API_KEY: OpenAI API authentication
  - MAX_BATCH_FILE_SIZE: 90 MB (OpenAI limit: 100 MB)
  - MAX_REQUESTS_PER_FILE: 49k (OpenAI limit: 50k)
  - POLL_INTERVAL: 600 seconds
```

### 2. Data Preparation Pipeline

**Purpose**: Convert cleaned data into streaming-ready WebDataset shards

**Flow**:
```
Cleaned CSV + Images
    │
    ├─ Load metadata
    ├─ Perform text preprocessing
    │
    ├─ Stratified Train/Val/Test split
    │  ├─ Train: 80% (45k samples)
    │  ├─ Val: 10% (5k samples)
    │  └─ Test: 10% (5k samples)
    │
    ├─ Create Mini-Train subset (5 shards)
    │
    ▼
Write WebDataset Shards
    │
    ├─ Resize & pad images
    ├─ Encode to JPEG
    ├─ Group by max_count (1000 samples/shard)
    ├─ Split by max_size (500 MB/shard)
    │
    ▼
S3 WebDataset Shards
  ├─ train/shard-000000.tar
  ├─ train/shard-000001.tar
  ├─ validation/shard-000000.tar
  └─ mini_train/shard-000000.tar
```

**Key Functions**:
- `write_dataset()`: Create tar shards
- `create_mini_train()`: Copy subset for testing
- `clean_metadata()`: Text preprocessing

**Benefits of WebDataset**:
- ✓ Efficient streaming from disk/S3
- ✓ Automatic batching in PyTorch
- ✓ Memory efficient (no loading entire dataset)
- ✓ Works with distributed training

### 3. Training Pipeline

**Purpose**: Fine-tune CLIP model with parameter-efficient techniques

**Architecture**:
```
                     ┌─ Vision Encoder ─┐
                     │   (ViT-B/32)     │
Input Images ────────┤                  ├──────┐
                     │   [LoRA Modules] │      │
                     └──────────────────┘      │
                                              ▼
                                        Projection
                                        Layer
                                        (trainable)
                                              ▲
                     ┌─ Text Encoder ──┐      │
                     │  (Transformer)  │      │
Input Text ─────────┤                 ├──────┘
                     │  [LoRA Modules] │
                     └──────────────────┘
```

**Training Scenarios** (Ablation Study):

| Scenario | Vision | Text | Projection | Use Case |
|----------|--------|------|-----------|----------|
| `zero_shot` | Frozen | Frozen | Frozen | Baseline comparison |
| `linear_probe` | Frozen | Frozen | Trainable | Fast convergence |
| `dual_lora` | LoRA | LoRA | Trainable | Best accuracy |
| `vision_lora` | LoRA | Frozen | Trainable | Domain adaptation |
| `text_lora` | Frozen | LoRA | Trainable | Language specialization |

**Optimization Strategy**:
```
Mixed Precision (AMP)
    ├─ fp16: Gradients & activations
    └─ fp32: Model parameters

Gradient Accumulation
    ├─ Effective batch = micro_batch × accumulation_steps
    └─ Example: 64 × 16 = 1024 effective

8-bit Optimizer (bnb.optim.AdamW8bit)
    ├─ Reduces memory by 4x
    └─ Maintains convergence rate

Learning Rate Schedule
    ├─ Warmup: Linear increase
    ├─ Main: Cosine annealing
    └─ Cycle: OneCycleLR
```

**Configuration** ([src/config.py](../src/config.py#L58-L73)):
```python
TrainingConfig:
  - BATCH_SIZE: 32
  - INITIAL_LR: 1e-4 (projections)
  - UNFROZEN_LR: 5e-6 (LoRA)
  - WEIGHT_DECAY: 0.01
  - EPOCHS: 5
```

**Checkpointing Strategy**:
```
Local Checkpoint ──┐
                   ├─ Save every N epochs
                   ├─ Upload to S3
                   └─ Resume on failure
```

### 4. Embedding & Evaluation Pipeline

**Embedding Generation**:
```
Dataset Items
    │
    ├─ Batch process (128 items/batch)
    │
    ├─ Base Model Embeddings
    │  └─ Shape: (N, 512)
    │
    ├─ Fine-tuned Model Embeddings
    │  └─ Shape: (N, 512)
    │
    ▼
Master Vector File (Parquet)
  ├─ image_id
  ├─ base_embedding (512-dim)
  ├─ finetuned_embedding (512-dim)
  └─ metadata
```

**Arena Evaluation Interface**:
```
┌─────────────────────────────────────┐
│         Query Image                 │
│  (Random furniture from test set)   │
├──────────────────┬──────────────────┤
│  Method A (TOP-4)|  Method B (TOP-4)│
│ Random order     | Random order     │
│                  |                  │
│ [A] Vote A       │ [B] Vote B       │
│ [N] Next         │ [Q] Quit         │
└──────────────────┴──────────────────┘
```

**Evaluation Metrics**:
- Win rate (A vs B)
- Tie rate
- Ranked results distribution
- Statistical significance (binomial test)

## Data Flow Details

### Cleaning Stage

```python
# Input: CSV with columns [id, name, image_path]
# Output: CSV with columns [..., clean_title, style, material, color, object_type]

# Processing:
for item in items:
    image_b64 = encode_image(item.image_path)
    prompt = create_vlm_prompt(item.name, image_b64)
    batch_request = {
        "custom_id": item.id,
        "body": {"model": "gpt-5-nano", "messages": prompt}
    }
    write_to_jsonl(batch_request)

# Result: JSONL files uploaded to OpenAI
# Polling status until completion
# Download results and merge with original CSV
```

### Preparation Stage

```python
# Input: Cleaned CSV + images
# Output: WebDataset shards in S3

# Processing:
for split in [train, val, test]:
    shard_writer = WebDatasetWriter(f"s3://bucket/{split}/")
    for sample in split_data[split]:
        image = load_and_resize_image(sample.image_path)
        record = {
            "__key__": f"{sample.id:09d}",
            "jpg": image_bytes,
            "txt": sample.prompt,
            "json": sample.metadata
        }
        shard_writer.write(record)
```

### Training Stage

```python
# Input: S3 WebDataset shards
# Output: Fine-tuned model checkpoint

# Processing:
for epoch in range(EPOCHS):
    for batch in train_loader:
        with autocast():
            outputs = model(
                input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                return_loss=True
            )
            loss = outputs.loss / grad_accum_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
```

### Evaluation Stage

```python
# Input: Test images + fine-tuned model
# Output: Arena evaluation results

# Processing:
for query_image in test_images:
    query_embedding = model.encode_image(query_image)
    
    # Retrieve top-K from both models
    base_topk = similarity_search(query_embedding, base_vectors, k=4)
    finetuned_topk = similarity_search(query_embedding, finetuned_vectors, k=4)
    
    # Display in random order (blind test)
    display_comparison(base_topk, finetuned_topk, randomize=True)
    
    # Record user vote
    save_vote(query_image, user_choice, model_order)
```

## State Management

### Cleaning State Machine

```
[Initial]
    │
    ├─ prepare() ──▶ [Prepared]
    │
    ├─ dispatch() ──▶ [In Progress]
    │                     │
    │                     ├─ monitor() ──▶ [Completed]
    │                     │
    │                     └─ failed() ──▶ [Failed]
    │
    └─ finalize() ──▶ [Finalized]
```

**State Storage** ([src/config.py](../src/config.py#L102)):
```json
{
  "batches": {
    "batch_123456": {
      "input_file": "batch_input_001.jsonl",
      "status": "completed",
      "output_file_id": "file_123",
      "local_output_path": "output_001.jsonl"
    }
  },
  "completed_files": ["cleaned_001.csv", "cleaned_002.csv"]
}
```

## Memory & Performance Optimization

### Training Optimizations

1. **Gradient Accumulation**:
   - Simulate larger batches with limited GPU memory
   - Formula: `effective_batch = micro_batch × accumulation_steps`

2. **Mixed Precision Training**:
   - ~2x speedup with minimal accuracy loss
   - Uses `torch.cuda.amp.autocast()`

3. **8-bit Optimizer**:
   - 4x memory reduction
   - Uses `bitsandbytes.optim.AdamW8bit`

4. **Parameter-Efficient Fine-tuning**:
   - LoRA adds only ~0.1% parameters
   - Reduces training memory by ~50%

### Inference Optimizations

1. **Batch Processing**:
   - Process 128 items per batch
   - Reduces I/O overhead

2. **GPU Inference**:
   - ~2.3ms per image on A100
   - ~10ms per image on RTX 4090

3. **Caching**:
   - Store embeddings in Parquet
   - No need to recompute

## Scalability

### Horizontal Scaling

**Multi-GPU Training**:
```python
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
# or
model = DistributedDataParallel(model)
```

**Distributed Cleaning**:
- Split CSV into multiple files
- Process batches in parallel

**S3 Streaming**:
- WebDataset automatically streams from S3
- No need to download entire dataset

### Vertical Scaling

**Memory Optimization**:
- Gradient checkpointing for large models
- Mixed precision reduces memory by 2x
- 8-bit optimization reduces by 4x

**Computation**:
- Use faster GPUs (A100 > RTX 4090 > T4)
- Enable TF32 for ~2x speedup on Ampere GPUs

## Error Recovery

### Automatic Checkpointing
```
Every epoch:
├─ Save model state
├─ Save optimizer state
├─ Upload to S3
└─ Enable resume on failure
```

### Batch API Resilience
```
Failed request:
├─ Retry with exponential backoff
├─ Save partial results
├─ Resume from checkpoint
└─ Report in state file
```

## Monitoring & Logging

### Weights & Biases Integration
```
Training Metrics:
├─ Loss (train/val)
├─ Learning rate
├─ Gradient statistics
├─ Model checkpoints
└─ Custom metrics
```

### Local Logging
```
Console Output:
├─ Progress bars (tqdm)
├─ Status messages
├─ Error messages
└─ Timing information
```

## References

- CLIP Paper: https://arxiv.org/abs/2103.14030
- LoRA Paper: https://arxiv.org/abs/2106.09685
- OpenAI Batch API: https://platform.openai.com/docs/guides/batch
- WebDataset: https://github.com/webdataset/webdataset
