# Version Compatibility Notes

## PyTorch Version Update (Important!)

**Updated:** PyTorch 2.1.0 → 2.2.0

### Why This Change?

The original setup specified:
- PyTorch 2.1.0 (October 2023)
- Transformers >=4.38 (February 2024)

**Problem:** Transformers 4.38+ requires PyTorch 2.2.0+ for the `torch.utils._pytree.register_pytree_node` API.

**Solution:** Upgraded to PyTorch 2.2.0 which:
- ✅ Still supports CUDA 11.8
- ✅ Compatible with transformers 4.38+
- ✅ Compatible with all other dependencies
- ✅ No breaking changes for the MoE training code

### Compatibility Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10 or 3.11 | Recommended |
| PyTorch | 2.2.0 | CUDA 11.8 |
| torchvision | 0.17.0 | Matches PyTorch 2.2.0 |
| torchaudio | 2.2.0 | Matches PyTorch 2.2.0 |
| transformers | >=4.38 | Requires PyTorch 2.2+ |
| torch-scatter | latest | Built for torch 2.2.0+cu118 |

### System Dependencies (Linux)

For RDKit rendering on headless systems:
```bash
apt-get install -y libxrender1 libxext6 libsm6 libfontconfig1
```

**Note:** RDKit rendering is optional - training works without these libraries.

### Migration from Original Setup

If you were using the original notebook with PyTorch 2.1.0:

1. **Clear your environment:**
   ```bash
   pip uninstall torch torchvision torchaudio transformers -y
   ```

2. **Run the updated notebook:**
   - The setup cells will install PyTorch 2.2.0 automatically
   - All other packages will be compatible

3. **Or use the setup script:**
   ```bash
   bash setup_runpod.sh
   ```

### Known Issues Resolved

✅ **Fixed:** `AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'`
- **Cause:** Transformers 4.38+ trying to use PyTorch 2.2+ API on PyTorch 2.1.0
- **Solution:** Upgraded to PyTorch 2.2.0

✅ **Fixed:** `ImportError: libXrender.so.1: cannot open shared object file`
- **Cause:** RDKit needs X11 libraries for rendering on headless systems
- **Solution:** Added automatic installation of system dependencies + graceful fallback

### Testing

Tested on:
- ✅ RunPod RTX 4090 (CUDA 11.8)
- ✅ Python 3.11
- ✅ Ubuntu 22.04 LTS

### References

- [PyTorch 2.2.0 Release Notes](https://github.com/pytorch/pytorch/releases/tag/v2.2.0)
- [Transformers Compatibility](https://github.com/huggingface/transformers#installation)
- [torch-scatter Installation](https://github.com/rusty1s/pytorch_scatter)
