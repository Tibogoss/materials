import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

# Add utils path for device management utilities
_utils_path = Path(__file__).parent.parent / 'utils'
if str(_utils_path) not in sys.path:
    sys.path.insert(0, str(_utils_path))

# Import device utilities
from device import get_model_device


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        print('Original embeddings:\n', out)
        return out


def get_expert_device(model):
    """
    Get device from expert model. Handles various model types:
    - Models with .device property (SMI-TED, SELFIES)
    - Models with nested submodules (MHG-GNN with .model attribute)
    - Standard nn.Module with parameters
    """
    # 1. Check if model has explicit device property
    if hasattr(model, 'device'):
        dev = model.device
        if isinstance(dev, torch.device):
            return dev
        elif isinstance(dev, str):
            return torch.device(dev)

    # 2. Check if model has a nested .model attribute with parameters
    if hasattr(model, 'model') and model.model is not None:
        try:
            return next(model.model.parameters()).device
        except (StopIteration, AttributeError):
            pass

    # 3. Check for _hf_model (SELFIES)
    if hasattr(model, '_hf_model') and model._hf_model is not None:
        try:
            return next(model._hf_model.parameters()).device
        except (StopIteration, AttributeError):
            pass

    # 4. Try standard parameters()
    if hasattr(model, 'parameters'):
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass

    # 5. Default to CPU
    return torch.device('cpu')


class Expert(nn.Module):
    def __init__(self, model, output_size, verbose=True, target_device=None):
        super().__init__()
        self.verbose = verbose
        self.model = model
        self.output_size = output_size
        self._target_device = target_device  # Explicitly set target device

    def set_device(self, device):
        """Set target device for outputs."""
        self._target_device = device

    def forward(self, x):
        # Determine target device
        if self._target_device is not None:
            device = self._target_device
        else:
            device = get_expert_device(self.model)

        # Check if input is empty and return an empty tensor of the appropriate shape
        if len(x) == 0:
            return torch.empty(size=(0, self.output_size), device=device)

        # Generate embeddings using the model's encode method
        out = self.model.encode(x)

        # Convert to tensor with CORRECT DEVICE
        if isinstance(out, pd.DataFrame):
            # Handle NaN values
            values = out.values.astype(np.float32)
            out = torch.tensor(values, dtype=torch.float32, device=device)
        elif isinstance(out, list):
            # Stack list of tensors and move to device
            if len(out) > 0 and isinstance(out[0], torch.Tensor):
                out = torch.stack(out, dim=0).to(device)
            else:
                out = torch.tensor(np.array(out), dtype=torch.float32, device=device)
        elif isinstance(out, torch.Tensor):
            out = out.to(device)
        elif isinstance(out, np.ndarray):
            out = torch.tensor(out, dtype=torch.float32, device=device)

        # Pad the embeddings to match the desired output size
        out = F.pad(out, pad=(0, self.output_size - out.shape[1], 0, 0), value=0)

        # Optionally print the embeddings if verbose mode is enabled
        if self.verbose:
            print(f'Original embeddings shape: {out.shape}')

        return out


class Net(nn.Module):
    def __init__(self, smiles_embed_dim, output_dim=2, dropout=0.2):
        super().__init__()
        self.desc_skip_connection = True 
        self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.GELU()
        self.final = nn.Linear(smiles_embed_dim, output_dim)

    def forward(self, smiles_emb):
        x_out = self.fc1(smiles_emb)
        x_out = self.dropout1(x_out)
        x_out = self.relu1(x_out)

        if self.desc_skip_connection is True:
            x_out = x_out + smiles_emb

        z = self.fc2(x_out)
        z = self.dropout2(z)
        z = self.relu2(z)
        if self.desc_skip_connection is True:
            z = self.final(z + x_out)
        else:
            z = self.final(z)

        return z