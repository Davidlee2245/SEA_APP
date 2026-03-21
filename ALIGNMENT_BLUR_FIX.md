# Alignment Blur Analysis & Fixes

## Problem Statement

After running alignment on contrast-enhanced images, the aligned output appears noticeably blurred compared to the preprocessed input. The grid pattern that was sharp after contrast enhancement becomes soft after alignment.

## Root Cause Analysis

### 1. **Bilinear Interpolation (Primary Cause)**
- **Issue**: Kornia's `warp_affine` uses **bilinear interpolation** by default, which causes blur during image warping.
- **Impact**: Bilinear interpolation averages neighboring pixels, reducing sharpness, especially for high-frequency features like grid patterns.
- **Evidence**: Grid lines that were crisp become soft/anti-aliased after alignment.

### 2. **Multiple Interpolation Steps**
- **Issue**: Images may be resampled multiple times:
  - During preprocessing (contrast enhancement may involve resampling)
  - During LoFTR feature detection (images >1024px are downsampled, then keypoints scaled back)
  - During alignment warping (bilinear interpolation)
- **Impact**: Each resampling step accumulates blur.

### 3. **Applying Alignment to Preprocessed Images**
- **Issue**: Alignment was being applied directly to contrast-enhanced images, which may have already been resampled during preprocessing.
- **Impact**: Double interpolation (preprocessing + alignment) compounds blur.

### 4. **Dtype Conversion and Quantization**
- **Issue**: Images are converted to float32 [0,1] for processing, then back to uint16/uint8, causing quantization errors.
- **Impact**: Subtle intensity loss and potential artifacts.

### 5. **No Interpolation Mode Control**
- **Issue**: No way to specify interpolation method (bilinear vs bicubic vs nearest).
- **Impact**: Always uses blurry bilinear interpolation.

## Implemented Fixes

### Fix 1: **Bicubic Interpolation Mode** ✅
- **Location**: `agents/aligner.py::apply_transform_kornia()`
- **Change**: 
  - Added `interpolation_mode` parameter ('bilinear', 'bicubic', 'nearest')
  - Replaced Kornia's `warp_affine` with `F.affine_grid` + `F.grid_sample` for interpolation control
  - Default set to `'bicubic'` for sharper results
- **Benefit**: Bicubic interpolation preserves sharpness better than bilinear, especially for grid patterns.

### Fix 2: **Estimate on Preprocessed, Apply to Raw** ✅
- **Location**: `api_server_extended.py::align_position()`
- **Change**:
  - Detect if raw images are available in cache
  - Estimate transform using preprocessed images (better feature detection due to contrast enhancement)
  - Apply transform to raw images (no preprocessing blur)
- **Benefit**: 
  - Best of both worlds: better feature detection (preprocessed) + sharp output (raw)
  - Eliminates double interpolation from preprocessing + alignment

### Fix 3: **Improved Dtype Preservation** ✅
- **Location**: `api_server_extended.py::align_position()`
- **Change**:
  - Track original dtype throughout the pipeline
  - Preserve bit depth when converting back from float32
  - Avoid unnecessary conversions
- **Benefit**: Reduces quantization errors and preserves image quality.

### Fix 4: **Pixel-to-Pixel Consistency Verification** ✅
- **Location**: `api_server_extended.py::align_position()`
- **Change**:
  - Verify that aligned image dimensions match original
  - Log warnings if shape mismatch detected
- **Benefit**: Catches dimension errors early, ensures no unexpected resizing.

### Fix 5: **Configuration Control** ✅
- **Location**: `agents/aligner.py::__init__()`
- **Change**:
  - Added `interpolation_mode` to aligner config
  - Can be set via API config: `'interpolation_mode': 'bicubic'`
- **Benefit**: Allows tuning interpolation method per use case.

## Technical Details

### Interpolation Modes Comparison

| Mode | Sharpness | Speed | Use Case |
|------|-----------|-------|----------|
| **Bilinear** | Low (blurry) | Fast | Default, smooth gradients |
| **Bicubic** | High (sharp) | Medium | **Recommended for microscopy** |
| **Nearest** | Highest (pixelated) | Fastest | Integer pixel shifts only |

### Transform Flow

**Before Fix:**
```
Raw → Preprocess (resample?) → Align (bilinear) → Blurry Output
```

**After Fix:**
```
Raw → Preprocess (for feature detection)
Raw → Align (bicubic, using transform from preprocessed) → Sharp Output
```

## Usage

### Default Behavior
The aligner now uses **bicubic interpolation** by default, which provides sharper results.

### Custom Interpolation Mode
To change interpolation mode, update the aligner config in `api_server_extended.py`:

```python
aligner_config = {
    # ... other config ...
    'interpolation_mode': 'bicubic'  # or 'bilinear', 'nearest'
}
```

### Verify Sharpness
After alignment, check:
1. Grid patterns remain sharp
2. No visible blur compared to preprocessed input
3. Terminal logs show: `"Applying transforms to raw images (preserving sharpness)"`

## Expected Results

After these fixes:
- ✅ Grid patterns remain sharp after alignment
- ✅ No noticeable blur compared to preprocessed input
- ✅ Better feature detection (using preprocessed for estimation)
- ✅ Sharp output (using raw images for warping)
- ✅ Preserved bit depth and image quality

## Testing

To verify the fixes work:
1. Load a position with visible grid patterns
2. Apply contrast enhancement (grid should be sharp)
3. Run alignment
4. Compare aligned image to preprocessed: **grid should remain sharp**

## Future Improvements

1. **Adaptive Interpolation**: Use bicubic for small shifts, nearest for integer pixel shifts
2. **Sub-pixel Refinement**: Use phase correlation for sub-pixel accuracy
3. **Multi-scale Alignment**: Estimate coarse transform on downsampled, refine on full resolution
4. **Sharpening Post-processing**: Optional unsharp mask after alignment to restore crispness

