# RunPod Setup Guide for MoE Caco-2 Training

This guide explains how to run the MoE_Caco2_Permeability notebook on a fresh RunPod GPU instance using `uv` for fast dependency management.

## Quick Start (Recommended)

The notebook now includes automatic setup! Just run all cells and it will configure everything for you.

```bash
# 1. Launch RunPod GPU instance (RTX 4090 / A6000 recommended)

# 2. Clone the materials repository
git clone https://github.com/IBM/materials.git
cd materials/models/mol_moe/notebooks

# 3. Launch Jupyter
jupyter notebook --allow-root --ip=0.0.0.0 --port=8888

# 4. Open MoE_Caco2_Permeability.ipynb and run all cells
```

**First run:** ~3-5 minutes (installs all packages with uv)
**Subsequent runs:** ~10-30 seconds (uses cached packages)

## Alternative: Command-Line Setup

If you prefer to set up the environment before opening the notebook:

```bash
# From materials/models/mol_moe/notebooks/
bash setup_runpod.sh
```

This runs the same installation steps as the notebook setup cells.

## What Changed?

### New Setup Cells (Automatic)

The notebook now includes 4 automatic setup cells at the beginning:

1. **Environment Detection** - Auto-detects platform, GPU, and configures all paths dynamically
2. **UV Installation** - Installs uv package manager if not present
3. **Dependency Installation** - Installs PyTorch 2.1.0 with CUDA 11.8 and all required packages
4. **Module Configuration** - Sets up Python import paths

### Path Fixes

All hardcoded relative paths have been replaced with dynamic paths:
- ✅ Data files: `DATA_DIR / 'train_Caco2_Permeability_Papp_AB.csv'`
- ✅ Tokenizer vocab: `EXPERTS_DIR / 'smi_ted_light' / 'bert_vocab_curated.txt'`
- ✅ Checkpoints: Saved to `notebooks/checkpoints/` directory

### Pre-Training Validation

A validation cell now checks before training:
- ✅ GPU availability and memory
- ✅ Data file exists and has correct columns
- ✅ Disk space (>10GB required)
- ✅ Model initialization
- ✅ Creates checkpoint directory

## Directory Structure

After running, your directory structure will be:

```
materials/
├── train_Caco2_Permeability_Papp_AB.csv
├── train_Caco2_Permeability_Efflux.csv
└── models/
    └── mol_moe/
        ├── moe/
        ├── experts/
        └── notebooks/
            ├── MoE_Caco2_Permeability.ipynb (MODIFIED)
            ├── pyproject.toml (NEW)
            ├── setup_runpod.sh (NEW)
            ├── README_RUNPOD.md (this file)
            └── checkpoints/ (created at runtime)
                ├── best_Caco2_Papp_AB_moe_model.pt
                ├── xgboost_Caco2_Papp_AB_model.json
                └── comparison_Caco2_Papp_AB.csv
```

## Requirements

- **Python:** 3.10 or 3.11 (recommended for best compatibility)
- **GPU:** CUDA-capable GPU with ≥16GB VRAM (RunPod RTX 4090 / A6000)
- **Disk:** ≥10GB free space
- **Internet:** Required for downloading pre-trained models from HuggingFace

## Installed Packages

The setup installs:
- PyTorch 2.2.0 with CUDA 11.8 (compatible with transformers 4.38+)
- torch-geometric, torch-scatter
- RDKit, SELFIES (chemistry libraries)
- Transformers, HuggingFace Hub (for pre-trained models)
- XGBoost, scikit-learn (ML tools)
- Pandas, NumPy, Matplotlib, Seaborn (data & visualization)

## Troubleshooting

### RDKit Rendering Error (libXrender.so.1)
If you see `ImportError: libXrender.so.1: cannot open shared object file`:

**Solution:** The notebook automatically tries to install system dependencies, but may need sudo access:
```bash
# If running setup script with sudo:
sudo bash setup_runpod.sh

# Or manually install:
sudo apt-get update
sudo apt-get install -y libxrender1 libxext6 libsm6 libfontconfig1
```

**Workaround:** The notebook will continue to work even without these libraries - you just won't see molecule images in DataFrames. Training is unaffected.

### Transformers/PyTorch Version Error
If you see `AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'`:

**Solution:** This is a version mismatch. The notebook now uses PyTorch 2.2.0 which is compatible with transformers 4.38+. Restart the kernel and run all cells from the beginning.

### GPU Not Detected
```bash
# Check GPU availability
nvidia-smi

# If GPU not showing, check CUDA installation
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Import Errors
Make sure you're running the notebook from the correct directory:
```bash
pwd  # Should show: .../materials/models/mol_moe/notebooks
```

### Out of Memory
Reduce batch size in the config cell:
```python
batch_size = 16  # Default is 32
```

### Slow Downloads
Pre-trained models are downloaded from HuggingFace on first run. This is normal and only happens once.

## Performance Tips

1. **Use uv caching:** After first run, uv caches all packages for instant reinstallation
2. **Save checkpoints:** Training saves best model automatically to `checkpoints/`
3. **Monitor GPU:** Use `nvidia-smi` to check GPU utilization during training
4. **Adjust epochs:** Default is 150 epochs (~2-3 hours on RTX 4090)

## Credits

This notebook uses the IBM Foundation Models for Materials (FM4M) framework:
- **MoE Framework:** Mixture-of-Experts combining SMI-TED, SELFIES-TED, and MHG-GNN
- **Migration to uv:** Fast, modern Python package management
- **RunPod Optimization:** Self-contained setup for cloud GPU instances

## Support

For issues with:
- **This setup:** Check [materials repository](https://github.com/IBM/materials) issues
- **UV package manager:** See [uv documentation](https://docs.astral.sh/uv/)
- **RunPod:** Check [RunPod documentation](https://docs.runpod.io/)
