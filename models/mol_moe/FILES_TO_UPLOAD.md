# Files to Upload to Remote Server

When deploying to a remote server, you need to upload these modified/new files to the IBM materials repository clone.

## Required Files (Device Fixes)

These files fix the CUDA/CPU device mismatch issues:

```
models/mol_moe/
├── experts/
│   ├── __init__.py                        # NEW - Package marker
│   ├── selfies_ted/
│   │   ├── __init__.py                    # NEW - Package marker
│   │   └── load.py                        # MODIFIED - Device handling, Column.copy() fix
│   ├── smi_ted_light/
│   │   ├── __init__.py                    # NEW - Package marker
│   │   └── load.py                        # MODIFIED - Device property, .cuda() removal
│   └── mhg_model/
│       └── (no changes needed)
├── moe/
│   ├── __init__.py                        # NEW - Package exports
│   ├── moe.py                             # MODIFIED - Device fixes, MoE.to() override
│   └── models.py                          # MODIFIED - Expert device detection
└── utils/
    ├── __init__.py                        # NEW - From previous work
    └── device.py                          # NEW - From previous work
```

## Training Scripts (NEW)

```
models/mol_moe/scripts/
├── train.py                               # NEW - CLI training script
├── inference.py                           # NEW - CLI inference script
├── pyproject.toml                         # NEW - UV dependencies
├── install_deps.sh                        # NEW - Dependency installer
└── README.md                              # NEW - Documentation
```

## Upload Methods

### Option 1: Git Push (Recommended)

```bash
# On local machine
cd /Users/tsouthiratn/Documents/NovaliX/OpenADMETv2/materials

git add models/mol_moe/experts/__init__.py
git add models/mol_moe/experts/selfies_ted/__init__.py
git add models/mol_moe/experts/selfies_ted/load.py
git add models/mol_moe/experts/smi_ted_light/__init__.py
git add models/mol_moe/experts/smi_ted_light/load.py
git add models/mol_moe/moe/__init__.py
git add models/mol_moe/moe/moe.py
git add models/mol_moe/moe/models.py
git add models/mol_moe/utils/
git add models/mol_moe/scripts/

git commit -m "Add MoL-MoE training scripts with device fixes"
git push origin main  # or your branch name

# On remote server
git clone <your-repo-url>
cd materials/models/mol_moe/scripts
./install_deps.sh
uv run --no-build-isolation train.py --data <data.csv> --smiles SMILES --target Activity
```

### Option 2: SCP Upload

```bash
# From local machine
cd /Users/tsouthiratn/Documents/NovaliX/OpenADMETv2/materials

# Upload experts directory changes
scp models/mol_moe/experts/__init__.py user@server:~/materials/models/mol_moe/experts/
scp models/mol_moe/experts/selfies_ted/__init__.py user@server:~/materials/models/mol_moe/experts/selfies_ted/
scp models/mol_moe/experts/selfies_ted/load.py user@server:~/materials/models/mol_moe/experts/selfies_ted/
scp models/mol_moe/experts/smi_ted_light/__init__.py user@server:~/materials/models/mol_moe/experts/smi_ted_light/
scp models/mol_moe/experts/smi_ted_light/load.py user@server:~/materials/models/mol_moe/experts/smi_ted_light/

# Upload moe directory changes
scp models/mol_moe/moe/__init__.py user@server:~/materials/models/mol_moe/moe/
scp models/mol_moe/moe/moe.py user@server:~/materials/models/mol_moe/moe/
scp models/mol_moe/moe/models.py user@server:~/materials/models/mol_moe/moe/

# Upload utils directory (entire folder)
scp -r models/mol_moe/utils user@server:~/materials/models/mol_moe/

# Upload scripts directory (entire folder)
scp -r models/mol_moe/scripts user@server:~/materials/models/mol_moe/
```

### Option 3: rsync (Cleanest)

```bash
# From local machine
rsync -avz --progress \
    models/mol_moe/experts/ \
    user@server:~/materials/models/mol_moe/experts/

rsync -avz --progress \
    models/mol_moe/moe/ \
    user@server:~/materials/models/mol_moe/moe/

rsync -avz --progress \
    models/mol_moe/utils/ \
    user@server:~/materials/models/mol_moe/utils/

rsync -avz --progress \
    models/mol_moe/scripts/ \
    user@server:~/materials/models/mol_moe/scripts/
```

## Verification on Server

After uploading, verify all files are present:

```bash
ssh user@server
cd ~/materials/models/mol_moe

# Check __init__.py files
ls -l experts/__init__.py
ls -l experts/selfies_ted/__init__.py
ls -l experts/smi_ted_light/__init__.py
ls -l moe/__init__.py

# Check modified files
ls -l experts/selfies_ted/load.py
ls -l experts/smi_ted_light/load.py
ls -l moe/moe.py
ls -l moe/models.py

# Check new directories
ls -l utils/
ls -l scripts/

# Test import
cd scripts
python3 -c "from experts.smi_ted_light.load import load_smi_ted; print('✓ Import successful')"
```

## Minimal File Set (If Space Constrained)

If you only want the essential files:

**Required:**
- `experts/__init__.py` (NEW)
- `experts/selfies_ted/__init__.py` (NEW)
- `experts/selfies_ted/load.py` (MODIFIED)
- `experts/smi_ted_light/__init__.py` (NEW)
- `experts/smi_ted_light/load.py` (MODIFIED)
- `moe/__init__.py` (NEW)
- `moe/moe.py` (MODIFIED)
- `moe/models.py` (MODIFIED)
- `scripts/train.py` (NEW)
- `scripts/inference.py` (NEW)
- `scripts/pyproject.toml` (NEW)
- `scripts/install_deps.sh` (NEW)

**Optional (but recommended):**
- `utils/` directory
- `scripts/README.md`
