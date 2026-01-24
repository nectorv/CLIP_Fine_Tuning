# Quick Start Guide

Get up and running with CLIP fine-tuning in 5 minutes.

## 1. Setup (1 min)

```bash
# Clone repository
git clone https://github.com/nectorv/CLIP_Fine_Tuning.git
cd CLIP_Fine_Tuning

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
OPENAI_API_KEY=sk-...
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
CUDA_VISIBLE_DEVICES=0
EOF
```

## 2. Data Cleaning (2 min setup)

**Option A: Using Orchestrator** (Recommended)
```bash
# Run full pipeline
python -m src.cleaning.main --all

# Or step-by-step
python -m src.cleaning.main --prepare
python -m src.cleaning.main --dispatch
python -m src.cleaning.main --monitor
python -m src.cleaning.main --finalize
```

**Option B: Manual Steps**
```bash
# Prepare batch files
python -m src.cleaning.main --prepare

# Check results manually at dashboard.openai.com

# Download results when complete
python -m src.cleaning.main --finalize
```

## 3. Data Preparation (1 min)

```bash
python -m src.preparing.main
# Creates WebDataset shards optimized for training
# Outputs to: data/webdataset_shards/
```

## 4. Train Model (1 min setup)

```bash
# Upload shards to S3 first
aws s3 sync data/webdataset_shards s3://my-bucket/data/

# Start training
python -m src.training.main \
  --s3_bucket my-bucket \
  --s3_prefix data \
  --scenario dual_lora \
  --epochs 3 \
  --lr 1e-4

# Monitor on Weights & Biases
# https://wandb.ai/
```

## 5. Evaluate Results

```bash
# Generate embeddings
python -m src.embeddings.embedder

# Launch evaluation GUI
python -m src.arena.app
# Controls: N=next, A/B=vote, Q=quit
```

## Minimal Example

For quick testing without full data:

```bash
# Create dummy data
mkdir -p data/raw/downloaded_images
python -c "
import pandas as pd
df = pd.DataFrame({
    'name': ['Sofa ' + str(i) for i in range(10)],
    'local_path': ['image_' + str(i) + '.jpg' for i in range(10)]
})
df.to_csv('data/raw/furniture_dataset_cleaned.csv', index=False)
"

# Create dummy images
python -c "
from PIL import Image
for i in range(10):
    Image.new('RGB', (256, 256)).save(f'data/raw/downloaded_images/image_{i}.jpg')
"

# Run cleaning with --smoke mode
python -m src.cleaning.main --prepare --smoke 5

# Continue with rest of pipeline...
```

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce `--micro_batch_size` to 32 |
| `S3 access denied` | Check AWS credentials in `.env` |
| `OpenAI API error` | Verify `OPENAI_API_KEY` in `.env` |
| `Module not found` | Run `pip install -r requirements.txt` |
| `No images found` | Check image paths in CSV relative to `data/raw` |

## Next Steps

- Read [README.md](../README.md) for detailed overview
- Check [API.md](API.md) for module documentation
- Review [docs/Architecture.md](Architecture.md) for system design
- Explore [notebooks/data_exploration.ipynb](../notebooks/data_exploration.ipynb)

## Configuration Quick Reference

**High-performance training:**
```bash
python -m src.training.main \
  --scenario dual_lora \
  --effective_batch_size 2048 \
  --micro_batch_size 128 \
  --lr 5e-5 \
  --epochs 5 \
  --warmup_steps 500
```

**Quick testing:**
```bash
python -m src.training.main \
  --scenario linear_probe \
  --epochs 1 \
  --effective_batch_size 256 \
  --mini_train
```

**Ablation study:**
```bash
for scenario in zero_shot linear_probe dual_lora vision_lora text_lora; do
  python -m src.training.main \
    --scenario $scenario \
    --epochs 3 \
    --wandb_project clip-ablation
done
```

## Getting Help

- **Check logs**: `wandb.ai` for training metrics
- **Debug mode**: Add `-vv` for verbose output
- **GitHub issues**: Submit with error traceback
- **API docs**: See [API.md](API.md)

## What's Next?

After your first successful run:

1. **Experiment with scenarios**: Try different LoRA configurations
2. **Monitor performance**: Track metrics on W&B dashboard
3. **Evaluate models**: Use Arena interface for human evaluation
4. **Optimize hyperparameters**: Adjust learning rate, batch size
5. **Deploy model**: Export to ONNX or TensorRT for production

Happy fine-tuning! 🚀
