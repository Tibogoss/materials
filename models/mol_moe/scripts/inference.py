#!/usr/bin/env python3
"""
MoL-MoE Inference Script

Run inference on SMILES using trained MoL-MoE models.
Designed to run as a background process on remote servers.

Usage:
    # Predict on single SMILES
    uv run scripts/inference.py --checkpoint checkpoints/best_model.pt --smiles "CCO"

    # Predict on CSV file
    uv run scripts/inference.py --checkpoint checkpoints/best_model.pt \
        --input data.csv --smiles-column SMILES --output predictions.csv

    # Predict from stdin (one SMILES per line)
    cat smiles.txt | uv run scripts/inference.py --checkpoint checkpoints/best_model.pt --stdin

    # Use XGBoost model instead of Net
    uv run scripts/inference.py --checkpoint checkpoints/best_model.pt \
        --xgboost checkpoints/xgboost_model.json --input data.csv

    # Run in background
    nohup uv run scripts/inference.py --checkpoint best.pt --input data.csv \
        --output preds.csv > inference.log 2>&1 &
"""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")


def setup_logging(log_file: Path = None, level: str = "INFO"):
    """Setup logging for both console and file output."""
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stderr)]  # Log to stderr, output to stdout
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
        description="Run inference with trained MoL-MoE model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model checkpoint (required)
    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True,
        help="Path to trained MoE model checkpoint (.pt file)"
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--smiles", "-s", type=str, nargs="+",
        help="SMILES string(s) to predict"
    )
    input_group.add_argument(
        "--input", "-i", type=str,
        help="Path to input CSV file"
    )
    input_group.add_argument(
        "--stdin", action="store_true",
        help="Read SMILES from stdin (one per line)"
    )

    # CSV options
    parser.add_argument(
        "--smiles-column", type=str, default="SMILES",
        help="Name of SMILES column in input CSV"
    )
    parser.add_argument(
        "--id-column", type=str, default=None,
        help="Name of ID column in input CSV (optional)"
    )

    # Output options
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file path. If not specified, prints to stdout."
    )
    parser.add_argument(
        "--format", "-f", type=str, choices=["csv", "json", "tsv"], default="csv",
        help="Output format"
    )

    # Model options
    parser.add_argument(
        "--xgboost", "-x", type=str, default=None,
        help="Path to XGBoost model (.json). If not specified, uses MoE+Net."
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=64,
        help="Batch size for inference"
    )

    # Device options
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified."
    )

    # Misc options
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output (only print results)"
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


def load_models(checkpoint_path: Path, device, logger):
    """Load MoE and Net models from checkpoint."""
    import torch  # Import here to ensure availability

    script_dir = Path(__file__).parent.resolve()
    mol_moe_root = script_dir.parent
    experts_dir = mol_moe_root / "experts"
    moe_dir = mol_moe_root / "moe"

    # Add to Python path
    sys.path.insert(0, str(mol_moe_root))
    sys.path.insert(0, str(experts_dir))
    sys.path.insert(0, str(moe_dir))

    logger.info("Loading expert models...")

    # Load expert models
    from experts.selfies_ted.load import SELFIES
    model_selfies = SELFIES()
    model_selfies.load()

    from experts.mhg_model.load import load as load_mhg
    mhg_gnn = load_mhg()

    from experts.smi_ted_light.load import load_smi_ted, MolTranBertTokenizer
    smi_ted = load_smi_ted()

    # Move to device
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
    tokenizer = MolTranBertTokenizer(str(vocab_path))

    # Load checkpoint to get args
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get('args', {})

    k = saved_args.get('k', 4)
    dropout = saved_args.get('dropout', 0.2)

    moe_model = MoE(
        input_size=768,
        output_size=2048,
        num_experts=12,
        models=models,
        tokenizer=tokenizer,
        tok_emb=smi_ted.encoder.tok_emb,
        k=k,
        noisy_gating=True,
        verbose=False
    ).to(device)

    net = Net(smiles_embed_dim=2048, dropout=dropout, output_dim=1)
    net = net.to(device)

    # Load weights
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    moe_model.load_state_dict(checkpoint['moe_state_dict'])
    net.load_state_dict(checkpoint['net_state_dict'])

    epoch = checkpoint.get('epoch', 'unknown')
    valid_loss = checkpoint.get('valid_loss', 'unknown')
    logger.info(f"Loaded model from epoch {epoch}, valid_loss={valid_loss}")

    return moe_model, net


def load_xgboost_model(xgboost_path: Path, logger):
    """Load XGBoost model from pickle file."""
    import pickle
    logger.info(f"Loading XGBoost model: {xgboost_path}")
    with open(xgboost_path, 'rb') as f:
        xgb_model = pickle.load(f)
    return xgb_model


def predict_batch(smiles_list, moe_model, net, xgb_model, device, batch_size, logger):
    """Run prediction on a batch of SMILES."""
    import torch
    import numpy as np

    moe_model.eval()
    if net is not None:
        net.eval()

    all_predictions = []
    all_valid_indices = []
    all_embeddings = []

    # Normalize SMILES
    normalized = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        norm_smi = normalize_smiles(smi)
        if norm_smi:
            normalized.append(norm_smi)
            valid_indices.append(i)
        else:
            logger.warning(f"Invalid SMILES at index {i}: {smi}")

    if not normalized:
        logger.error("No valid SMILES to process")
        return [None] * len(smiles_list)

    # Process in batches
    logger.info(f"Processing {len(normalized)} valid SMILES in batches of {batch_size}...")

    with torch.no_grad():
        for start in range(0, len(normalized), batch_size):
            end = min(start + batch_size, len(normalized))
            batch = normalized[start:end]

            embeddings, _ = moe_model(batch, verbose=False)
            all_embeddings.append(embeddings.cpu())

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)

    # Predict
    if xgb_model is not None:
        predictions = xgb_model.predict(all_embeddings.numpy())
    else:
        with torch.no_grad():
            predictions = net(all_embeddings.to(device)).squeeze().cpu().numpy()

    # Handle single prediction case
    if predictions.ndim == 0:
        predictions = np.array([predictions.item()])

    # Map predictions back to original indices
    results = [None] * len(smiles_list)
    for idx, pred in zip(valid_indices, predictions):
        results[idx] = float(pred)

    return results


def main():
    args = parse_args()

    # Setup logging
    log_level = "WARNING" if args.quiet else args.log_level
    logger = setup_logging(level=log_level)

    logger.info("="*60)
    logger.info("MoL-MoE Inference Script")
    logger.info("="*60)

    # Import dependencies
    import torch
    import numpy as np
    import pandas as pd

    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Using device: {device}")

    # Validate checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load models
    moe_model, net = load_models(checkpoint_path, device, logger)

    # Load XGBoost if specified
    xgb_model = None
    if args.xgboost:
        xgb_path = Path(args.xgboost)
        if not xgb_path.exists():
            logger.error(f"XGBoost model not found: {xgb_path}")
            sys.exit(1)
        xgb_model = load_xgboost_model(xgb_path, logger)
        net = None  # Use XGBoost instead of Net

    # Get input SMILES
    smiles_list = []
    ids = []
    input_df = None  # Store original dataframe to preserve all columns

    if args.smiles:
        # From command line
        smiles_list = args.smiles
        ids = [f"mol_{i}" for i in range(len(smiles_list))]
        logger.info(f"Input: {len(smiles_list)} SMILES from command line")

    elif args.stdin:
        # From stdin
        logger.info("Reading SMILES from stdin...")
        for line in sys.stdin:
            line = line.strip()
            if line:
                smiles_list.append(line)
        ids = [f"mol_{i}" for i in range(len(smiles_list))]
        logger.info(f"Input: {len(smiles_list)} SMILES from stdin")

    elif args.input:
        # From CSV file
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)

        logger.info(f"Reading from: {input_path}")
        input_df = pd.read_csv(input_path)

        if args.smiles_column not in input_df.columns:
            logger.error(f"SMILES column '{args.smiles_column}' not found. Available: {list(input_df.columns)}")
            sys.exit(1)

        smiles_list = input_df[args.smiles_column].tolist()

        if args.id_column and args.id_column in input_df.columns:
            ids = input_df[args.id_column].tolist()
        else:
            ids = [f"mol_{i}" for i in range(len(smiles_list))]

        logger.info(f"Input: {len(smiles_list)} SMILES from CSV")

    if not smiles_list:
        logger.error("No SMILES to process")
        sys.exit(1)

    # Run predictions
    predictions = predict_batch(
        smiles_list, moe_model, net, xgb_model, device, args.batch_size, logger
    )

    # Prepare output
    if input_df is not None:
        # CSV input: preserve all original columns and add prediction
        output_df = input_df.copy()
        output_df['prediction'] = predictions
        results = output_df.to_dict('records')
    else:
        # Command line or stdin: create simple output
        results = []
        for i, (smi, pred, mol_id) in enumerate(zip(smiles_list, predictions, ids)):
            results.append({
                'id': mol_id,
                'smiles': smi,
                'prediction': pred
            })

    # Count valid predictions
    valid_count = sum(1 for pred in predictions if pred is not None)
    logger.info(f"Predictions completed: {valid_count}/{len(predictions)} valid")

    # Output results
    # Use dataframe directly if available, otherwise create from results
    df_out = output_df if input_df is not None else pd.DataFrame(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.format == "json":
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        elif args.format == "tsv":
            df_out.to_csv(output_path, sep='\t', index=False)
        else:  # csv
            df_out.to_csv(output_path, index=False)

        logger.info(f"Results saved to: {output_path}")

    else:
        # Print to stdout
        if args.format == "json":
            print(json.dumps(results, indent=2))
        elif args.format == "tsv":
            df_out.to_csv(sys.stdout, sep='\t', index=False)
        else:  # csv
            df_out.to_csv(sys.stdout, index=False)

    logger.info("Inference completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
