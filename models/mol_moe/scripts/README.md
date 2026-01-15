# MoL-MoE Training & Inference Scripts

Production-ready command-line scripts for training and inference with MoL-MoE models.

## Quick Start

### Option 1: Automatic (Recommended)

```bash
cd scripts

# First time: install dependencies
./install_deps.sh

# Then run training
uv run --no-build-isolation train.py \
    --data ../../../your_data.csv \
    --smiles SMILES \
    --target "Your Target Column"
```

### Option 2: Direct Run (if pyproject.toml fix works)

```bash
cd scripts

uv run train.py \
    --data ../../../your_data.csv \
    --smiles SMILES \
    --target "Your Target Column"
```

## Installation

### Install UV (one-time)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### Install Dependencies

**Method 1: Using install script (recommended)**
```bash
./install_deps.sh
```

**Method 2: Manual**
```bash
uv pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu118

uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

uv pip install torch-geometric matplotlib==3.9.2 numpy==1.26.4 pandas \
    scikit-learn rdkit datasets huggingface-hub transformers==4.44.2 \
    selfies tqdm xgboost==2.1.3 seaborn
```

## Training

### Basic Training

```bash
uv run --no-build-isolation train.py \
    --data train.csv \
    --smiles SMILES \
    --target Activity
```

### Advanced Options

```bash
# Train with custom hyperparameters
uv run --no-build-isolation train.py \
    --data train.csv \
    --smiles SMILES \
    --target "Caco-2 Permeability" \
    --epochs 200 \
    --batch-size 16 \
    --lr 5e-5 \
    --k 4 \
    --train-xgboost

# Resume training from checkpoint
uv run --no-build-isolation train.py \
    --data train.csv \
    --smiles SMILES \
    --target Activity \
    --resume checkpoints/best_model.pt

# Run in background
nohup uv run --no-build-isolation train.py \
    --data train.csv \
    --smiles SMILES \
    --target Activity \
    > training.log 2>&1 &

# Check progress
tail -f training.log
```

### Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | Required | Path to CSV file |
| `--smiles` | Required | SMILES column name |
| `--target` | Required | Target column name |
| `--epochs` | 150 | Number of epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--dropout` | 0.2 | Dropout rate |
| `--k` | 4 | Experts to activate |
| `--train-xgboost` | False | Also train XGBoost |
| `--resume` | None | Resume from checkpoint |
| `--output-dir` | ./checkpoints | Output directory |

## Inference

### Predict on CSV File

```bash
uv run --no-build-isolation inference.py \
    --checkpoint checkpoints/best_model.pt \
    --input molecules.csv \
    --smiles-column SMILES \
    --output predictions.csv
```

### Predict Single SMILES

```bash
uv run --no-build-isolation inference.py \
    --checkpoint checkpoints/best_model.pt \
    --smiles "CCO" "c1ccccc1" "CC(=O)O"
```

### Predict from stdin

```bash
cat smiles.txt | uv run --no-build-isolation inference.py \
    --checkpoint checkpoints/best_model.pt \
    --stdin > predictions.csv
```

### Use XGBoost Model

```bash
uv run --no-build-isolation inference.py \
    --checkpoint checkpoints/best_model.pt \
    --xgboost checkpoints/xgboost_model.json \
    --input data.csv
```

## Troubleshooting

### "torch-scatter build failed"

Use the install script:
```bash
./install_deps.sh
```

Then always use `--no-build-isolation` flag:
```bash
uv run --no-build-isolation train.py ...
```

### CUDA out of memory

Reduce batch size:
```bash
uv run --no-build-isolation train.py ... --batch-size 16
```

### Check GPU availability

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Output Files

After training, you'll find in `checkpoints/`:
- `best_<model_name>_moe_model.pt` - Best MoE+Net checkpoint
- `xgboost_<model_name>_model.json` - XGBoost model (if `--train-xgboost`)
- `results_<model_name>.csv` - Performance metrics
- `train_<model_name>_<timestamp>.log` - Training log

## Examples

### Example 1: Caco-2 Permeability

```bash
uv run --no-build-isolation train.py \
    --data train_Caco2_Permeability_Papp_AB.csv \
    --smiles SMILES \
    --target "Caco-2 Permeability Papp A>B" \
    --epochs 150 \
    --train-xgboost
```

### Example 2: Binary Classification

```bash
uv run --no-build-isolation train.py \
    --data bace.csv \
    --smiles smiles \
    --target Class \
    --epochs 100
```

## Remote Server Usage

```bash
# SSH into server
ssh user@server

# Install UV (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Clone repo
git clone https://github.com/IBM/materials
cd materials/models/mol_moe/scripts

# Install dependencies
./install_deps.sh

# Upload data file
# (use scp from local: scp data.csv user@server:~/materials/models/mol_moe/)

# Run in background
nohup uv run --no-build-isolation train.py \
    --data ../data.csv \
    --smiles SMILES \
    --target Activity \
    > training.log 2>&1 &

# Disconnect and let it run
exit

# Later: check progress via SSH
ssh user@server
cd materials/models/mol_moe/scripts
tail -f training.log
```
