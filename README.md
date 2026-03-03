# CLIP Fine-Tuning for Semantic Furniture Search

A comprehensive, production-ready system for fine-tuning OpenAI's CLIP model for semantic search in furniture product catalogs.

## What's This About?

This repository provides an end-to-end pipeline for fine-tuning Vision-Language models (specifically CLIP) on domain-specific furniture data. The system includes:

- **Data Cleaning Pipeline**: Automated batch cleaning via OpenAI's GPT-5 nano (Batch API)
- **Data Preparation**: Efficient WebDataset-based sharding for large-scale training
- **Fine-Tuning**: Multiple fine-tuning strategies using LoRA and parameter-efficient techniques
- **Embedding Generation**: Batch computation of semantic embeddings for search
- **Evaluation Interface**: Interactive arena-based human evaluation tools

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU training)
- AWS S3 access (for data storage and model checkpointing)
- OpenAI API key (for batch cleaning operations)

### Installation

```bash
git clone <repository-url>
cd CLIP_Fine_Tuning
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
MODEL_NAME=openai/clip-vit-base-patch32
CUDA_VISIBLE_DEVICES=0
```

## Project Structure

```
├── src/
│   ├── config.py              # Centralized configuration
│   ├── cleaning/              # Data cleaning pipeline
│   │   ├── api_client.py      # OpenAI Batch API wrapper
│   │   ├── cleaner.py         # Batch cleaning logic
│   │   ├── orchestrator.py    # Pipeline orchestration
│   │   ├── processor.py       # Data processing utilities
│   │   ├── prompts.py         # LLM prompts
│   │   ├── schema.py          # Pydantic schemas
│   │   └── state_manager.py   # State management
│   ├── preparing/             # Data preparation
│   │   ├── image_processing.py
│   │   ├── text_processing.py
│   │   ├── sharding.py        # WebDataset sharding
│   │   └── main.py
│   ├── training/              # Model training
│   │   ├── data.py            # Data loaders
│   │   ├── model.py           # Model architecture
│   │   ├── utils.py           # Training utilities
│   │   └── main.py
│   └── embeddings/            # Embedding generation
│       ├── embedder.py        # Embedding generator
│       └── utils.py
├── notebooks/
│   └── data_exploration.ipynb
├── requirements.txt
└── README.md
```

## Usage

### 1. Data Cleaning

Clean raw furniture product data using GPT-5 nano VLM:

```bash
# Prepare batch files
python -m src.cleaning.main --prepare

# Upload to OpenAI and start processing
python -m src.cleaning.main --dispatch

# Monitor batch progress
python -m src.cleaning.main --monitor

# Download and finalize results
python -m src.cleaning.main --finalize
```

### 2. Data Preparation

Prepare cleaned data for training:

```bash
python -m src.preparing.main
```

This generates WebDataset shards optimized for streaming training:
- Splits data into train/validation/test sets
- Creates mini-train subset for quick testing
- Uploads to S3 for distributed training

### 3. Model Training

Fine-tune CLIP with various strategies:

```bash
python -m src.training.main \
  --s3_bucket my-bucket \
  --s3_prefix data/train \
  --scenario dual_lora \
  --epochs 10 \
  --lr 1e-4 \
  --effective_batch_size 1024
```

Supported scenarios:
- `zero_shot`: No fine-tuning (baseline)
- `linear_probe`: Only projection layers trainable
- `dual_lora`: LoRA on both encoders + unfrozen projections
- `vision_lora`: LoRA on vision encoder only
- `text_lora`: LoRA on text encoder only

### 4. Embedding Generation

Generate embeddings for the entire dataset:

```bash
python -m src.embeddings/embedder.py
```

Outputs master vector file for semantic search operations.
