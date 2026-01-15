#!/bin/bash
# Install dependencies for MoL-MoE training
# This handles the torch-scatter build dependency issue

set -e

echo "Installing PyTorch and dependencies..."

# Install torch first (required for torch-scatter build)
uv pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install torch-scatter from prebuilt wheel
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

# Install torch-geometric
uv pip install torch-geometric

# Install remaining dependencies
uv pip install \
    matplotlib==3.9.2 \
    numpy==1.26.4 \
    pandas \
    scikit-learn \
    rdkit \
    datasets \
    huggingface-hub \
    transformers==4.44.2 \
    selfies \
    tqdm \
    xgboost==2.1.3 \
    seaborn

echo "✓ All dependencies installed successfully!"
echo ""
echo "Now you can run training with:"
echo "  uv run --no-build-isolation train.py --data <your_data.csv> --smiles SMILES --target <target_column>"
