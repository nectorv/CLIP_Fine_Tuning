# Documentation Index

Welcome to the CLIP Fine-Tuning project documentation. This guide helps you navigate all available resources.

## Quick Navigation

### Getting Started 🚀
Start here if you're new to the project:
1. **[QUICKSTART.md](QUICKSTART.md)** - Get running in 5 minutes
2. **[README.md](../README.md)** - Project overview and features
3. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design overview

### For Developers 👨‍💻
Deep dive into implementation details:
1. **[API.md](API.md)** - Complete API reference
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Component architecture & data flow
3. **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guidelines
4. **[Notebooks](../notebooks/)** - Interactive examples

### For Users 👤
Run and use the system:
1. **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide
2. **[README.md](../README.md)** - Usage instructions
3. **[TROUBLESHOOTING.md](#troubleshooting)** - Common issues

---

## Documentation Map

### README.md
**Main project documentation**
- Project overview
- Installation instructions
- Basic usage examples
- Acknowledgments
- License information

### QUICKSTART.md
**Fast-track onboarding**
- 5-minute setup
- Step-by-step pipeline execution
- Minimal example for testing
- Quick troubleshooting

### ARCHITECTURE.md
**System design & internals**
- Component architecture
- Data flow diagrams
- Pipeline descriptions
- State management
- Performance optimizations
- Scalability considerations

### API.md
**Complete API reference**
- Module documentation
- Class descriptions
- Function signatures
- Code examples
- Usage patterns
- Performance tips

### CONTRIBUTING.md
**Contribution guidelines**
- Development setup
- Code style standards
- Testing procedures
- PR process
- Issue templates
- Release process

---

## Learning Path

### Path 1: Quick User (⏱️ 10 minutes)
1. Read [QUICKSTART.md](QUICKSTART.md)
2. Run minimal example
3. Try training script
4. Check results in Arena

### Path 2: Full User (⏱️ 1 hour)
1. Read [README.md](../README.md) - Overview
2. Follow [QUICKSTART.md](QUICKSTART.md) - Setup
3. Review [ARCHITECTURE.md](ARCHITECTURE.md) - System design
4. Run complete pipeline
5. Evaluate with Arena interface

### Path 3: Developer (⏱️ 2 hours)
1. Complete Path 2
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) - Deep dive
3. Study [API.md](API.md) - Implementation details
4. Read [CONTRIBUTING.md](CONTRIBUTING.md) - Development setup
5. Explore [Notebooks](../notebooks/)
6. Start contributing

### Path 4: Advanced Researcher (⏱️ 4 hours)
1. Complete Path 3
2. Read all architecture sections
3. Study optimization techniques
4. Run ablation studies
5. Modify for your use case
6. Contribute improvements

---

## Feature Overview

### Data Cleaning
📄 **Docs**: [ARCHITECTURE.md#1-data-cleaning-pipeline](ARCHITECTURE.md#1-data-cleaning-pipeline) | [API.md#srccleaning](API.md#srccleaning-apikey)

Extract structured metadata from raw furniture data using GPT-5 nano VLM via OpenAI Batch API.

```bash
python -m src.cleaning.main --all
```

### Data Preparation
📄 **Docs**: [ARCHITECTURE.md#2-data-preparation-pipeline](ARCHITECTURE.md#2-data-preparation-pipeline) | [API.md#srcpreparingsharding](API.md#srcpreparingsharding)

Convert cleaned data into WebDataset shards optimized for streaming training.

```bash
python -m src.preparing.main
```

### Model Training
📄 **Docs**: [ARCHITECTURE.md#3-training-pipeline](ARCHITECTURE.md#3-training-pipeline) | [API.md#srctrainingmain](API.md#srctrainingmain)

Fine-tune CLIP with LoRA using multiple training scenarios for ablation studies.

```bash
python -m src.training.main --scenario dual_lora --epochs 5
```

### Embedding Generation
📄 **Docs**: [ARCHITECTURE.md#4-embedding--evaluation-pipeline](ARCHITECTURE.md#4-embedding--evaluation-pipeline) | [API.md#srcembeddingsembedder](API.md#srcembeddingsembedder)

Generate embeddings for semantic search operations.

```bash
python -m src.embeddings.embedder
```

### Evaluation Arena
📄 **Docs**: [ARCHITECTURE.md#4-embedding--evaluation-pipeline](ARCHITECTURE.md#4-embedding--evaluation-pipeline) | [API.md#srcarenapp](API.md#srcarenapp)

Interactive human-in-the-loop comparison interface with blind testing.

```bash
python -m src.arena.app
```

---

## Common Tasks

### Setup & Installation
- [QUICKSTART.md - Section 1](QUICKSTART.md#1-setup-1-min)
- [README.md - Installation](../README.md#installation)

### Running the Pipeline
- [QUICKSTART.md - All sections](QUICKSTART.md)
- [README.md - Usage](../README.md#usage)

### Understanding Architecture
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [API.md - Data Flow](API.md#data-flow)

### Adding New Features
- [CONTRIBUTING.md - Getting Started](CONTRIBUTING.md#getting-started)
- [CONTRIBUTING.md - Development Workflow](CONTRIBUTING.md#development-workflow)
- [ARCHITECTURE.md - Relevant component](ARCHITECTURE.md)

### Debugging Issues
- [QUICKSTART.md - Troubleshooting](QUICKSTART.md#troubleshooting-quick-fixes)
- [README.md - Troubleshooting](../README.md#troubleshooting)
- [GitHub Issues](https://github.com/nectorv/CLIP_Fine_Tuning/issues)

### Optimizing Performance
- [ARCHITECTURE.md - Memory & Performance](ARCHITECTURE.md#memory--performance-optimization)
- [ARCHITECTURE.md - Scalability](ARCHITECTURE.md#scalability)
- [README.md - Performance Benchmarks](../README.md#performance-benchmarks)

### Contributing Code
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CONTRIBUTING.md - Pull Request Process](CONTRIBUTING.md#pull-request-process)

---

## File Organization

```
CLIP_Fine_Tuning/
├── README.md                          # Main documentation
├── docs/
│   ├── QUICKSTART.md                 # This file
│   ├── ARCHITECTURE.md               # System design
│   ├── API.md                        # API reference
│   ├── CONTRIBUTING.md               # Contribution guide
│   └── INDEX.md                      # Documentation index
├── src/
│   ├── config.py                     # Configuration
│   ├── cleaning/                     # Data cleaning
│   ├── preparing/                    # Data preparation
│   ├── training/                     # Model training
│   ├── embeddings/                   # Embedding generation
│   └── arena/                        # Evaluation interface
├── notebooks/
│   └── data_exploration.ipynb        # Interactive notebook
└── requirements.txt                  # Dependencies
```

---

## Troubleshooting Quick Links

| Issue | Solution |
|-------|----------|
| CUDA out of memory | [README.md#troubleshooting](../README.md#troubleshooting) |
| Module not found | [QUICKSTART.md](QUICKSTART.md#troubleshooting-quick-fixes) |
| S3 access errors | [QUICKSTART.md](QUICKSTART.md#troubleshooting-quick-fixes) |
| Model training issues | [ARCHITECTURE.md#training-pipeline](ARCHITECTURE.md#3-training-pipeline) |
| Arena display problems | [README.md#troubleshooting](../README.md#troubleshooting) |

---

## Resources

### Official Documentation
- [CLIP Paper](https://arxiv.org/abs/2103.14030)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [OpenAI Batch API Docs](https://platform.openai.com/docs/guides/batch)
- [WebDataset GitHub](https://github.com/webdataset/webdataset)

### Related Projects
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PEFT Library](https://github.com/huggingface/peft)

### Community
- [Hugging Face Community](https://huggingface.co/community)
- [PyTorch Discussion Forums](https://discuss.pytorch.org/)
- [GitHub Issues](https://github.com/nectorv/CLIP_Fine_Tuning/issues)

---

## Version Info

- **Project Version**: 1.0.0
- **Last Updated**: January 2024
- **Python Version**: 3.9+
- **PyTorch Version**: 2.0+
- **CUDA Version**: 11.8+

---

## Getting Help

### Before Asking
1. Check [QUICKSTART.md](QUICKSTART.md#troubleshooting-quick-fixes)
2. Search [existing issues](https://github.com/nectorv/CLIP_Fine_Tuning/issues)
3. Review relevant documentation section

### How to Ask
1. **For questions**: Open a GitHub Discussion (tag: question)
2. **For bugs**: Open an issue with reproduction steps
3. **For features**: Open an issue to discuss first
4. **For help**: Check documentation then ask in Discussions

### Reporting Issues
Include:
- Python & CUDA versions
- Error message (full traceback)
- Minimal reproduction code
- Environment details

---

## Next Steps

- 👤 **New User?** Start with [QUICKSTART.md](QUICKSTART.md)
- 👨‍💻 **Developer?** Read [CONTRIBUTING.md](CONTRIBUTING.md)
- 🔍 **Deep Dive?** Study [ARCHITECTURE.md](ARCHITECTURE.md)
- 📚 **API Details?** Refer to [API.md](API.md)

---

**Happy learning!** 🚀

For questions or feedback about documentation, please open an issue on GitHub.
