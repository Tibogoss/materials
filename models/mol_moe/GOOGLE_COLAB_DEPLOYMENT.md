# Google Colab Deployment Guide

## Quick File Replacement Checklist

After `git clone https://github.com/IBM/materials.git`, replace these files:

### ✅ Files to Replace (5 files)

| File Path | Changes | Critical |
|-----------|---------|----------|
| `models/mol_moe/moe/moe.py` | 60 lines modified | ⚠️ YES |
| `models/mol_moe/moe/models.py` | 25 lines modified | ⚠️ YES |
| `models/mol_moe/experts/selfies_ted/load.py` | 35 lines modified | ⚠️ YES |
| `models/mol_moe/experts/smi_ted_light/load.py` | 80 lines modified | ⚠️ YES |
| `models/mol_moe/experts/mhg_model/load.py` | No changes needed | ℹ️ NO (optional) |

### ✅ Files to Add (2 new files)

| File Path | Purpose |
|-----------|---------|
| `models/mol_moe/utils/__init__.py` | Empty init file |
| `models/mol_moe/utils/device.py` | Device utilities (140 lines) |

### ✅ Notebook to Use

| File Path | Description |
|-----------|-------------|
| `models/mol_moe/notebooks/MoE_Production_Template.ipynb` | Production notebook (47 cells) |

---

## Colab Commands

```bash
# In Colab cell:
!git clone https://github.com/IBM/materials.git
%cd materials/models/mol_moe

# Upload your 5 modified files to replace originals
# Upload your 2 new utils files
# Upload the production notebook

# Open MoE_Production_Template.ipynb
# Edit cells 11-12 for your data
# Run all cells
```

---

## What to Edit in Notebook

### Cell 11 - Data Configuration
```python
DATA_FILE = DATA_DIR / 'YOUR_FILE.csv'
SMILES_COLUMN = 'SMILES'           # Your SMILES column name
TARGET_COLUMN = 'YOUR_TARGET'       # Your target column name
MODEL_NAME = 'MyModel'              # Name for checkpoints
```

### Cell 12 - Hyperparameters (Optional)
```python
batch_size = 32        # Reduce if OOM (try 16)
learning_rate = 1e-4
epochs = 150
```

---

## Expected Runtime (on Colab with GPU)

- **First run setup**: ~3-5 minutes (dependency installation)
- **Subsequent runs setup**: ~30 seconds (cached)
- **Training**: Depends on dataset size and epochs
  - Small dataset (1000 samples, 50 epochs): ~15-20 minutes
  - Medium dataset (2000 samples, 150 epochs): ~1-2 hours

---

## Troubleshooting

### Issue: "CUDA not available"
**Solution**: Enable GPU in Colab (Runtime → Change runtime type → GPU)

### Issue: "Module not found"
**Solution**: Verify utils/ folder was created and files uploaded correctly

### Issue: "Data file not found"
**Solution**: Upload your CSV to Colab, update `DATA_FILE` path in cell 11

### Issue: "Out of memory (OOM)"
**Solution**: Reduce `batch_size` in cell 12 (try 16 or 8)

---

## Success Indicators

✅ Cell 2: Environment detected (Platform, GPU available)
✅ Cell 4: PyTorch 2.2.0 with CUDA 11.8 installed
✅ Cell 9: MoE and Net modules importable
✅ Cell 20: All pre-training validation checks passed
✅ Cell 24: Training completes without device errors
✅ Cell 27: Test metrics calculated successfully
✅ Cell 32: Comparison shows both models working

---

## File Sizes (Reference)

- `moe/moe.py`: ~12 KB (original: ~11 KB)
- `moe/models.py`: ~4 KB (original: ~3 KB)
- `experts/selfies_ted/load.py`: ~4 KB (original: ~3 KB)
- `experts/smi_ted_light/load.py`: ~24 KB (original: ~23 KB)
- `utils/device.py`: ~7 KB (new)
- `MoE_Production_Template.ipynb`: ~39 KB (new)

---

## Critical Device Fixes Included

1. ✅ Training labels moved to device (fixes GPU/CPU mismatch in loss)
2. ✅ Expert outputs created on correct device (fixes DataFrame tensor issue)
3. ✅ EmbeddingNet tokenizer outputs moved to device
4. ✅ SparseDispatcher batch_index aligned
5. ✅ SELFIES multiprocessing disabled (fixes CUDA fork errors)
6. ✅ SMI-TED uses device property (cleaner than hardcoded .cuda())
7. ✅ Net model moved to device in notebook
8. ✅ train() function accepts device parameter

All 8 critical device issues are fixed! 🎉
