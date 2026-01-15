#!/usr/bin/env python3
"""
MoL-MoE Training Script

Train MoL-MoE (Mixture of Experts) models on custom CSV datasets.
Designed to run as a background process on remote servers.

Usage:
    uv run scripts/train.py --data train.csv --smiles SMILES --target Activity

    # With custom hyperparameters
    uv run scripts/train.py --data train.csv --smiles SMILES --target Activity \
        --epochs 200 --batch-size 16 --lr 5e-5

    # Run in background
    nohup uv run scripts/train.py --data train.csv --smiles SMILES --target Activity \
        > training.log 2>&1 &
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# Configure logging
def setup_logging(log_file: Path = None, level: str = "INFO"):
    """Setup logging for both console and file output."""
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MoL-MoE model on custom CSV data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--data", "-d", type=str, required=True,
        help="Path to CSV file with training data"
    )
    parser.add_argument(
        "--smiles", "-s", type=str, required=True,
        help="Name of SMILES column in CSV"
    )
    parser.add_argument(
        "--target", "-t", type=str, required=True,
        help="Name of target column in CSV"
    )

    # Model configuration
    parser.add_argument(
        "--name", "-n", type=str, default=None,
        help="Model name for checkpoints (default: derived from target column)"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="./checkpoints",
        help="Directory for saving checkpoints and logs"
    )

    # Hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=150,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr", "--learning-rate", type=float, default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2,
        help="Dropout rate"
    )
    parser.add_argument(
        "--k", type=int, default=4,
        help="Number of experts to activate per sample"
    )

    # Data split ratios
    parser.add_argument(
        "--train-ratio", type=float, default=0.70,
        help="Training set ratio"
    )
    parser.add_argument(
        "--valid-ratio", type=float, default=0.15,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15,
        help="Test set ratio"
    )

    # Training options
    parser.add_argument(
        "--train-xgboost", action="store_true",
        help="Also train XGBoost on MoE embeddings"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified."
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    return parser.parse_args()


def normalize_smiles(smi, canonical=True, isomeric=False):
    """Normalize SMILES string."""
    from rdkit import Chem
    try:
        normalized = Chem.MolToSmiles(
            Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
        )
    except:
        normalized = None
    return normalized


def main():
    args = parse_args()

    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    mol_moe_root = script_dir.parent
    experts_dir = mol_moe_root / "experts"
    moe_dir = mol_moe_root / "moe"

    # Add to Python path
    sys.path.insert(0, str(mol_moe_root))
    sys.path.insert(0, str(experts_dir))
    sys.path.insert(0, str(moe_dir))

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model name
    model_name = args.name or args.target.replace(" ", "_").replace("/", "_")

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"train_{model_name}_{timestamp}.log"
    logger = setup_logging(log_file, args.log_level)

    logger.info("="*60)
    logger.info("MoL-MoE Training Script")
    logger.info("="*60)
    logger.info(f"Data file: {args.data}")
    logger.info(f"SMILES column: {args.smiles}")
    logger.info(f"Target column: {args.target}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*60)

    # Import dependencies
    logger.info("Importing dependencies...")
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from tqdm import tqdm

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    logger.info("Loading data...")
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")

    # Validate columns
    if args.smiles not in df.columns:
        logger.error(f"SMILES column '{args.smiles}' not found in CSV. Available: {list(df.columns)}")
        sys.exit(1)
    if args.target not in df.columns:
        logger.error(f"Target column '{args.target}' not found in CSV. Available: {list(df.columns)}")
        sys.exit(1)

    # Normalize SMILES
    logger.info("Normalizing SMILES...")
    df['canon_smiles'] = df[args.smiles].apply(normalize_smiles)
    original_count = len(df)
    df = df.dropna(subset=['canon_smiles', args.target])
    logger.info(f"Removed {original_count - len(df)} invalid entries, {len(df)} remaining")

    # Target statistics
    logger.info(f"Target statistics: mean={df[args.target].mean():.4f}, std={df[args.target].std():.4f}")

    # Split data
    train_df, temp_df = train_test_split(df, test_size=(1-args.train_ratio), random_state=args.seed)
    valid_size = args.valid_ratio / (args.valid_ratio + args.test_ratio)
    valid_df, test_df = train_test_split(temp_df, test_size=(1-valid_size), random_state=args.seed)

    logger.info(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    # Prepare data
    X_train = train_df['canon_smiles'].tolist()
    y_train = torch.tensor(train_df[args.target].values, dtype=torch.float32)

    X_valid = valid_df['canon_smiles'].tolist()
    y_valid = torch.tensor(valid_df[args.target].values, dtype=torch.float32)

    X_test = test_df['canon_smiles'].tolist()
    y_test = torch.tensor(test_df[args.target].values, dtype=torch.float32)

    # Load expert models
    logger.info("Loading expert models...")

    from experts.selfies_ted.load import SELFIES
    logger.info("  Loading SELFIES-TED...")
    model_selfies = SELFIES()
    model_selfies.load()

    from experts.mhg_model.load import load as load_mhg
    logger.info("  Loading MHG-GNN...")
    mhg_gnn = load_mhg()

    from experts.smi_ted_light.load import load_smi_ted, MolTranBertTokenizer
    logger.info("  Loading SMI-TED...")
    smi_ted = load_smi_ted()

    logger.info("Expert models loaded successfully")

    # Move expert models to device
    logger.info(f"Moving expert models to {device}...")
    smi_ted.to(device)
    model_selfies.to(device)
    mhg_gnn.to(device)

    # Initialize MoE
    logger.info("Initializing MoE model...")
    from moe import MoE
    from models import Net

    models = [
        smi_ted, smi_ted, smi_ted, smi_ted,
        model_selfies, model_selfies, model_selfies, model_selfies,
        mhg_gnn, mhg_gnn, mhg_gnn, mhg_gnn
    ]

    vocab_path = experts_dir / 'smi_ted_light' / 'bert_vocab_curated.txt'
    if not vocab_path.exists():
        logger.error(f"Vocab file not found: {vocab_path}")
        sys.exit(1)
    tokenizer = MolTranBertTokenizer(str(vocab_path))

    moe_model = MoE(
        input_size=768,
        output_size=2048,
        num_experts=12,
        models=models,
        tokenizer=tokenizer,
        tok_emb=smi_ted.encoder.tok_emb,
        k=args.k,
        noisy_gating=True,
        verbose=False
    ).to(device)

    net = Net(smiles_embed_dim=2048, dropout=args.dropout, output_dim=1)
    net.apply(smi_ted._init_weights)
    net = net.to(device)

    logger.info(f"MoE parameters: {sum(p.numel() for p in moe_model.parameters()):,}")
    logger.info(f"Net parameters: {sum(p.numel() for p in net.parameters()):,}")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_valid_loss = float('inf')

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger.info(f"Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            moe_model.load_state_dict(checkpoint['moe_state_dict'])
            net.load_state_dict(checkpoint['net_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_valid_loss = checkpoint.get('valid_loss', float('inf'))
            logger.info(f"Resumed from epoch {start_epoch}, best_valid_loss={best_valid_loss:.4f}")
        else:
            logger.warning(f"Checkpoint not found: {resume_path}, starting fresh")

    # Training setup
    loss_fn = nn.MSELoss()
    params = list(moe_model.parameters()) + list(net.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=10, verbose=False
    )

    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    # Training loop
    logger.info("="*60)
    logger.info("Starting training...")
    logger.info(f"Epochs: {start_epoch} to {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info("="*60)

    train_losses = []
    valid_losses = []

    for epoch in range(start_epoch, args.epochs):
        # Training
        moe_model.train()
        net.train()
        epoch_loss = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=args.log_level == "WARNING"
        )

        for (x, y) in progress_bar:
            y = y.to(device)

            optim.zero_grad()

            embd, aux_loss = moe_model(x)
            y_hat = net(embd).squeeze()

            loss = loss_fn(y_hat, y)
            total_loss = loss + aux_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optim.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        moe_model.eval()
        net.eval()
        with torch.no_grad():
            valid_embd, _ = moe_model(X_valid, verbose=False)
            valid_preds = net(valid_embd).squeeze()
            valid_loss = loss_fn(valid_preds.cpu(), y_valid).item()

        valid_losses.append(valid_loss)

        # Learning rate scheduling
        old_lr = optim.param_groups[0]['lr']
        scheduler.step(valid_loss)
        new_lr = optim.param_groups[0]['lr']
        if new_lr != old_lr:
            logger.info(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")

        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            checkpoint_path = output_dir / f'best_{model_name}_moe_model.pt'
            torch.save({
                'epoch': epoch,
                'moe_state_dict': moe_model.state_dict(),
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'valid_loss': valid_loss,
                'args': vars(args),
            }, checkpoint_path)
            logger.info(f"Epoch {epoch+1}: New best model saved (valid_loss={valid_loss:.4f})")

        # Periodic logging
        if (epoch + 1) % 10 == 0 or epoch == start_epoch:
            logger.info(
                f"Epoch {epoch+1}/{args.epochs}: "
                f"train_loss={avg_train_loss:.4f}, valid_loss={valid_loss:.4f}, "
                f"best={best_valid_loss:.4f}"
            )

    logger.info("="*60)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_valid_loss:.4f}")
    logger.info("="*60)

    # Load best model for evaluation
    checkpoint_path = output_dir / f'best_{model_name}_moe_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    moe_model.load_state_dict(checkpoint['moe_state_dict'])
    net.load_state_dict(checkpoint['net_state_dict'])

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    moe_model.eval()
    net.eval()

    with torch.no_grad():
        test_embd, _ = moe_model(X_test, verbose=False)
        test_preds = net(test_embd).squeeze()
        test_preds_np = test_preds.cpu().numpy()
        y_test_np = y_test.numpy()

    rmse = np.sqrt(mean_squared_error(y_test_np, test_preds_np))
    mae = mean_absolute_error(y_test_np, test_preds_np)
    r2 = r2_score(y_test_np, test_preds_np)

    logger.info("="*60)
    logger.info(f"TEST SET RESULTS - {model_name}")
    logger.info("="*60)
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE:  {mae:.4f}")
    logger.info(f"R²:   {r2:.4f}")
    logger.info("="*60)

    # Save results
    results = {
        'model_name': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'best_valid_loss': best_valid_loss,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
    }

    results_path = output_dir / f'results_{model_name}.csv'
    pd.DataFrame([results]).to_csv(results_path, index=False)
    logger.info(f"Results saved to: {results_path}")

    # Train XGBoost if requested
    if args.train_xgboost:
        logger.info("="*60)
        logger.info("Training XGBoost on MoE embeddings...")
        logger.info("="*60)

        from xgboost import XGBRegressor

        with torch.no_grad():
            xgb_train, _ = moe_model(X_train, verbose=False)
            xgb_valid, _ = moe_model(X_valid, verbose=False)
            xgb_test, _ = moe_model(X_test, verbose=False)

        xgb_train = xgb_train.cpu().numpy()
        xgb_valid = xgb_valid.cpu().numpy()
        xgb_test = xgb_test.cpu().numpy()

        y_train_np = y_train.numpy()
        y_valid_np = y_valid.numpy()

        xgb_model = XGBRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=args.seed,
            early_stopping_rounds=50,
            eval_metric='rmse'
        )

        xgb_model.fit(
            xgb_train, y_train_np,
            eval_set=[(xgb_valid, y_valid_np)],
            verbose=100
        )

        xgb_preds = xgb_model.predict(xgb_test)
        xgb_rmse = np.sqrt(mean_squared_error(y_test_np, xgb_preds))
        xgb_mae = mean_absolute_error(y_test_np, xgb_preds)
        xgb_r2 = r2_score(y_test_np, xgb_preds)

        logger.info("="*60)
        logger.info(f"XGBoost TEST SET RESULTS - {model_name}")
        logger.info("="*60)
        logger.info(f"RMSE: {xgb_rmse:.4f}")
        logger.info(f"MAE:  {xgb_mae:.4f}")
        logger.info(f"R²:   {xgb_r2:.4f}")
        logger.info("="*60)

        # Save XGBoost model using pickle (more reliable for sklearn wrapper)
        import pickle
        xgb_model_path = output_dir / f'xgboost_{model_name}_model.pkl'
        with open(xgb_model_path, 'wb') as f:
            pickle.dump(xgb_model, f)
        logger.info(f"XGBoost model saved to: {xgb_model_path}")

    logger.info("="*60)
    logger.info("All tasks completed successfully!")
    logger.info(f"Checkpoints saved to: {output_dir}")
    logger.info("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
