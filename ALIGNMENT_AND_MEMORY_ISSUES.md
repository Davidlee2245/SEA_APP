# 🔍 Alignment & Memory Issues Explanation

## Issue 1: Insufficient Matches for Image Alignment

### **Problem:**
```
WARNING | Insufficient matches for r01c01f02p01-ch2sk1fk1fl1. Using identity transform.
```

### **Root Cause:**
The alignment code requires **at least 4 matches** to estimate a transformation:
```python
if matches is None or len(matches) < 4:
    self.logger.warning(f"Insufficient matches for {channel_name}. Using identity transform.")
```

**Why LoFTR might return < 4 matches:**
1. **Images already well-aligned**: If channels are already registered, there are no distinctive features to match
2. **Low contrast / poor SNR**: Fluorescence images with low signal may not have enough detectable features
3. **Different structures**: If channels show completely different biological structures, there's nothing to match
4. **LoFTR model mismatch**: The `'indoor'` pretrained model is optimized for indoor scenes, not fluorescence microscopy
5. **Image quality**: Blurry, noisy, or overexposed images reduce feature detection

### **Current Behavior:**
- When < 4 matches: Uses **identity transform** (no alignment)
- This is a **fallback**, not an error
- The pipeline continues processing

### **Solutions:**

#### **Option 1: Lower the minimum match threshold** (Quick fix)
```yaml
# config/config.yaml
aligner:
  min_matches: 2  # Lower from 4 to 2 (allows simpler transforms)
```

#### **Option 2: Use fluorescence-optimized model** (Better)
- Switch from LoFTR `'indoor'` to a model trained on microscopy data
- Or use KeyNet+HardNet (already implemented as fallback)

#### **Option 3: Improve image preprocessing** (Best)
- Apply denoising before alignment
- Enhance contrast (CLAHE, histogram equalization)
- Use anchor channel with highest SNR

#### **Option 4: Add diagnostic logging**
- Log number of detected features per channel
- Log match confidence scores
- Visualize matched keypoints for debugging

---

## Issue 2: Memory Issues on RTX 3060 Ti

### **Problem:**
RTX 3060 Ti has **8 GB VRAM**, which is limited for running multiple deep learning models simultaneously.

### **Observations:**
1. **NUMA warnings** (lines 588-616): These are **informational**, not errors
   - TensorFlow/XLA initialization messages
   - Not related to actual memory problems

2. **CUDA memory usage (RTX 3060 Ti - 8 GB total):**
   - LoFTR (transformer-based): ~2-4 GB per image pair
   - StarDist: ~1-2 GB
   - Multiple models loaded: ~3-6 GB total
   - **System overhead**: ~1-2 GB
   - **Available for models**: ~6-7 GB (tight!)

3. **Image size:**
   - Original: 1080x1080
   - Downsampled to: 1024x1024 for LoFTR (already implemented to save memory)
   - Still memory-intensive for transformer models

### **Potential Issues (RTX 3060 Ti - 8 GB VRAM):**

#### **A. Multiple models in memory simultaneously**
- LoFTR model: ~500 MB
- StarDist model: ~200 MB
- KeyNet+HardNet (fallback): ~100 MB
- **Total: ~800 MB just for models**
- **Problem**: With only 8 GB total, running both LoFTR + StarDist simultaneously can cause OOM

#### **B. Batch processing**
- Processing multiple channels sequentially
- Each channel loads full model into GPU
- Memory not freed between channels
- **Problem**: Memory accumulates across channels

#### **C. StarDist memory spikes**
- StarDist can use significant memory during inference
- Especially with large images (1080x1080)
- **Problem**: Combined with LoFTR, can exceed 8 GB limit

#### **D. RTX 3060 Ti limitations**
- **8 GB VRAM is tight** for modern transformer models
- LoFTR alone can use 3-4 GB
- StarDist adds another 1-2 GB
- **Total: 4-6 GB just for inference** (before overhead)

### **Solutions:**

#### **Option 1: Clear CUDA cache between operations** (Already implemented)
```python
if self.device == "cuda":
    torch.cuda.empty_cache()
```

#### **Option 2: Reduce image size for LoFTR** (Already implemented)
```python
max_size = 1024  # Downsample to 1024x1024
```

#### **Option 3: Use CPU fallback for LoFTR** (Recommended for RTX 3060 Ti)
```yaml
# config/config.yaml
aligner:
  use_cpu_for_loftr: true  # Force CPU for LoFTR (slower but saves GPU memory)
```
**For RTX 3060 Ti**: This is recommended to avoid OOM errors. LoFTR on CPU is slower but stable.

#### **Option 4: Process channels sequentially with cleanup** (Recommended for RTX 3060 Ti)
- Unload LoFTR after alignment phase
- Load StarDist only when needed
- Clear cache between phases
- **For RTX 3060 Ti**: This prevents both models from being in memory simultaneously

#### **Option 5: Use mixed precision (FP16)**
- Reduce memory usage by 50%
- Requires model modification

#### **Option 6: Check actual GPU memory**
```python
# Add diagnostic logging
import torch
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

---

## Recommended Fixes for RTX 3060 Ti (8 GB VRAM)

### **Immediate Actions:**

1. **Monitor GPU memory usage:**
   ```bash
   # Run this in a separate terminal while pipeline is running
   watch -n 1 nvidia-smi
   ```
   **Expected**: Memory should stay below 7 GB. If it exceeds, you'll see OOM errors.

2. **Add diagnostic logging for matches:**
   ```python
   # In aligner.py, detect_and_match_loftr()
   if mkpts0.shape[1] < 4:
       self.logger.warning(f"Only {mkpts0.shape[1]} matches found (need 4). "
                          f"Image quality or alignment may be poor.")
       return None
   ```

2. **Add memory monitoring:**
   ```python
   # Before/after LoFTR inference
   if self.device == "cuda":
       mem_before = torch.cuda.memory_allocated() / 1e9
       # ... LoFTR inference ...
       mem_after = torch.cuda.memory_allocated() / 1e9
       self.logger.debug(f"LoFTR memory: {mem_before:.2f} GB -> {mem_after:.2f} GB")
   ```

3. **Lower minimum matches threshold:**
   ```python
   # In aligner.py
   min_matches = self.config.get('min_matches', 4)  # Make configurable
   if matches is None or len(matches) < min_matches:
   ```

### **Configuration Updates for RTX 3060 Ti:**

```yaml
# config/config.yaml
aligner:
  min_matches: 2  # Lower threshold (allows 2-point transforms)
  max_image_size: 1024  # Already implemented (saves memory)
  clear_cache_between_channels: true  # New option (recommended for 8 GB)
  use_cpu_fallback_on_oom: true  # New option (auto-fallback if OOM)
  # Consider: use_cpu_for_loftr: true  # Force CPU to save GPU memory
```

**RTX 3060 Ti Specific Recommendations:**
- ✅ Keep `max_image_size: 1024` (already set)
- ✅ Enable `clear_cache_between_channels: true`
- ⚠️ Consider `use_cpu_for_loftr: true` if you see OOM errors
- ⚠️ Process samples one at a time (don't batch)

---

## Questions to Investigate

1. **Why are matches insufficient?**
   - Check image SNR (Inspector phase)
   - Visualize matched keypoints
   - Compare anchor vs. moving channel contrast

2. **Is GPU memory actually the issue?** (RTX 3060 Ti - 8 GB)
   - Check `nvidia-smi` during pipeline execution
   - Monitor memory usage per phase
   - Check if OOM errors occur (not just warnings)
   - **Expected**: Memory usage should stay below 7 GB to avoid OOM
   - **If > 7 GB**: Consider CPU fallback or model unloading

3. **Are images already aligned?**
   - If channels are pre-aligned, identity transform is correct
   - Check if alignment is actually needed

---

## Next Steps

1. ✅ Add diagnostic logging for match counts
2. ✅ Add memory monitoring
3. ✅ Make minimum matches configurable
4. ✅ Add option to use CPU fallback for LoFTR
5. ⏳ Test with different image sizes
6. ⏳ Compare LoFTR vs. KeyNet+HardNet performance

---

**Status:** Issues identified, solutions proposed

**Date:** January 26, 2026

