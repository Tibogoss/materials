#!/bin/bash
# RunPod setup script for MoE Caco-2 training
# This script provides an alternative to the in-notebook setup cells
# Usage: bash setup_runpod.sh

set -e  # Exit on error

echo "=================================================="
echo "MoE Caco-2 Permeability Training - RunPod Setup"
echo "=================================================="
echo ""

# Install system dependencies for RDKit (Linux only)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "[0/5] Installing system dependencies for RDKit..."
    if command -v apt-get &> /dev/null; then
        apt-get update -qq || echo "⚠️  Could not update package list (may need sudo)"
        apt-get install -y -qq libxrender1 libxext6 libsm6 libfontconfig1 || echo "⚠️  Could not install some packages (may need sudo)"
        echo "      ✓ System dependencies installed"
    else
        echo "      ⚠️  apt-get not found, skipping system packages"
    fi
fi

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"
if [ "$python_version" != "3.10" ]; then
    echo "⚠️  Warning: Python 3.10 recommended, found $python_version"
    echo "   Some dependencies may have compatibility issues"
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo ""
    echo "[1/5] Installing uv..."
    pip install -q uv
    echo "      ✓ uv installed"
else
    echo ""
    echo "[1/5] uv already installed ($(uv --version))"
fi

# Install PyTorch with CUDA 11.8
echo ""
echo "[2/5] Installing PyTorch 2.1.0 with CUDA 11.8..."
uv pip install --python python3 \
    --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
echo "      ✓ PyTorch with CUDA 11.8 installed"

# Install torch-scatter
echo ""
echo "[3/5] Installing torch-scatter..."
uv pip install --python python3 \
    --find-links https://data.pyg.org/whl/torch-2.1.0+cu118.html \
    torch-scatter
echo "      ✓ torch-scatter installed"

# Install remaining dependencies
echo ""
echo "[4/5] Installing remaining dependencies..."
uv pip install --python python3 \
    torch-geometric>=2.3.1 \
    matplotlib==3.9.2 \
    'numpy>=1.26.1,<2.0.0' \
    'pandas>=1.5.3' \
    'scikit-learn>=1.5.0' \
    'rdkit>=2024.3.5' \
    'datasets>=2.13.1' \
    huggingface-hub \
    'transformers>=4.38' \
    'selfies>=2.1.0' \
    'tqdm>=4.66.4' \
    xgboost==2.0.0 \
    seaborn
echo "      ✓ All dependencies installed"

# Verify installation
echo ""
echo "=================================================="
echo "Verifying installation..."
echo "=================================================="

python3 -c "
import torch
import sys

print(f'Python: {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=================================================="
echo "✓ Setup complete!"
echo ""
echo "To start training:"
echo "  1. cd materials/models/mol_moe/notebooks"
echo "  2. jupyter notebook MoE_Caco2_Permeability.ipynb"
echo "  3. Run all cells"
echo "=================================================="
