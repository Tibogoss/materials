import os
import sys
import torch
import selfies as sf  # selfies>=2.1.1
import pickle
import pandas as pd
import numpy as np
from datasets import Dataset
from rdkit import Chem
from transformers import AutoTokenizer, AutoModel
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()


class SELFIES(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self._hf_model = None  # Will be set in load()
        self.tokenizer = None
        self.invalid = []
        self._device = None  # Track device

    @property
    def model(self):
        """Access the HuggingFace model."""
        return self._hf_model

    @model.setter
    def model(self, value):
        """Set the HuggingFace model."""
        self._hf_model = value

    @property
    def device(self):
        """Get current device of the model."""
        if self._device is not None:
            return self._device
        if self._hf_model is not None:
            try:
                return next(self._hf_model.parameters()).device
            except StopIteration:
                pass
        return torch.device('cpu')

    def to(self, device):
        """Move model to device."""
        device = torch.device(device) if isinstance(device, str) else device
        self._device = device
        if self._hf_model is not None:
            self._hf_model = self._hf_model.to(device)
        return self

    def cuda(self, device=None):
        """Move model to CUDA."""
        if device is None:
            device = torch.device('cuda')
        return self.to(device)

    def cpu(self):
        """Move model to CPU."""
        return self.to(torch.device('cpu'))

    def get_selfies(self, smiles_list):
        self.invalid = []
        spaced_selfies_batch = []
        for i, smiles in enumerate(smiles_list):
            try:
                selfies = sf.encoder(smiles.rstrip())
            except:
                try:
                    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles.rstrip()))
                    selfies = sf.encoder(smiles)
                except:
                    selfies = "[]"
                    self.invalid.append(i)

            spaced_selfies_batch.append(selfies.replace('][', '] ['))

        return spaced_selfies_batch


    def get_embedding(self, selfies):
        encoding = self.tokenizer(selfies["selfies"], return_tensors='pt', max_length=128, truncation=True, padding='max_length')

        # Fix: Move to model device
        device = self.device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = self._hf_model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        model_output = outputs.last_hidden_state

        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        model_output = sum_embeddings / sum_mask

        # Clean up encoding dict
        del encoding['input_ids']
        del encoding['attention_mask']

        # Move to CPU for dataset processing
        encoding["embedding"] = model_output.detach().cpu().numpy().tolist()

        return encoding


    def load(self, checkpoint="bart-2908.pickle"):
        """
            inputs :
                   checkpoint (pickle object)
        """

        self.tokenizer = AutoTokenizer.from_pretrained("ibm/materials.selfies-ted")
        self._hf_model = AutoModel.from_pretrained("ibm/materials.selfies-ted")


    # TODO: remove `use_gpu` argument in validation pipeline
    def encode(self, smiles_list=[], use_gpu=False, return_tensor=False):
        """
            inputs :
                   checkpoint (pickle object)
            :return: embedding
        """
        selfies = self.get_selfies(smiles_list)
        selfies_df = pd.DataFrame(selfies,columns=["selfies"])
        data = Dataset.from_pandas(selfies_df)

        # Fix: Disable multiprocessing (CRITICAL for CUDA compatibility)
        # num_proc=None prevents CUDA fork issues
        embedding = data.map(self.get_embedding, batched=True, num_proc=None, batch_size=128)

        # Fix: Use list() instead of .copy() - HuggingFace datasets Column doesn't have copy()
        emb = np.asarray(list(embedding["embedding"]))

        for idx in self.invalid:
            emb[idx] = np.nan
            print("Cannot encode {0} to selfies and embedding replaced by NaN".format(smiles_list[idx]))

        if return_tensor:
            return torch.tensor(emb)
        return pd.DataFrame(emb)
