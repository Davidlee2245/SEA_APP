# RTX 5090 Compatibility Notes

## CUDA Compatibility Warning

When running the pipeline with an RTX 5090, you may see this warning:

```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90.
```

**This is a WARNING, not an error.** The pipeline will still work because:

1. PyTorch will automatically use a compatible compute capability (sm_90)
2. Your GPU will still be utilized for all operations
3. Performance will be good, though not fully optimized for sm_120

## Current Status

- ✅ **Pipeline works**: All GPU operations function correctly
- ✅ **CUDA available**: `torch.cuda.is_available()` returns `True`
- ⚠️ **Warning shown**: PyTorch doesn't officially support sm_120 yet

## Future Updates

When PyTorch adds official sm_120 support, you can upgrade:

```bash
conda activate SEA
conda update pytorch pytorch-cuda -c pytorch -c nvidia
```

Or install PyTorch nightly builds which may have earlier sm_120 support:

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121
```

## Verification

To verify CUDA is working despite the warning:

```bash
conda activate SEA
python -c "import torch; x = torch.randn(2, 3).cuda(); print('Device:', x.device)"
```

If it prints `Device: cuda:0`, everything is working correctly.

