# SEA — Revised Analysis Plan
## Separation of Processed and Raw Image Roles

---

## 1. Overview

All analysis in SEA operates on two parallel image tracks that must never be mixed for quantitative purposes:

| Track | Source | Permitted Uses |
|-------|--------|---------------|
| **Display track** | Preprocessed (8-bit, contrast-enhanced, denoised) | Visualization, manual annotation guidance |
| **Analysis track** | Raw aligned images (original bit-depth, unmodified intensity) | Feature extraction, thresholding, segmentation input, all reported values |

The preprocessing pipeline (`preprocess_engine.py`) converts 16-bit raw data to 8-bit and applies operations such as CLAHE, background subtraction, gamma correction, and unsharp masking. These transformations alter pixel intensity in non-linear, non-invertible ways. Any number derived from a processed image — thresholds, mean intensities, RF training features — is therefore not scientifically meaningful as a quantitative measurement.

---

## 2. Current State vs. Required State

### 2.1 What the code does today (problems to fix)

| Location | Current behavior | Problem |
|----------|-----------------|---------|
| `exosome_segment()` — SAM path | Prefers processed image stages | SAM receives processed pixel values; image used for segmentation is not the raw source |
| `exosome_segment()` — blob path | Prefers raw but falls back to processed | Inconsistent; processed fallback contaminates threshold |
| `exosome_segment()` — RF path | Passes `image_array` from whichever stage was resolved | RF features extracted from processed pixels when raw is not explicitly enforced |
| `random_forest_segmentation.py::extract_features()` | Accepts any image; normalizes to [0,1] | No enforcement that the input is raw; processed image silently accepted |
| `exosome_export()` — overlay | Resolves image from processed stages first | Overlay drawn on processed image; acceptable for display but not labeled as such |
| `exosome_export()` — CSV | Exports only area, centroid, bbox, score | No raw-channel intensity values; perimeter/circularity missing from CSV |

### 2.2 Required state after implementation

- All three detection methods (SAM, Blob, RF) receive only raw aligned images as segmentation input.
- The display canvas shows the processed image; annotation coordinates are collected in processed-image pixel space.
- Because preprocessing does not rescale or resample (only intensity transforms), pixel coordinates are identical between processed and raw images. Annotation coordinates require no geometric remapping.
- RF feature vectors are extracted exclusively from raw pixel values.
- All quantitative outputs (intensity, area, perimeter, circularity, colocalization) are measured on raw images using the detected masks.

---

## 3. Two-Track Architecture

```
Raw aligned TIFFs  ─────────────────────────────────────────────────┐
  (16-bit, per channel)                                              │
        │                                                            │
        ├─── preprocess_engine ──► Processed 8-bit images           │
        │        (display track)         │                           │
        │                                ▼                           │
        │                    [ Display canvas / UI ]                 │
        │                    [ User annotation GUI ]                 │
        │                           │                                │
        │              Annotation pixel coords                       │
        │              (same coordinate space)                       │
        │                           │                                ▼
        │                           └──────────────────► RF training & prediction
        │                                                 (features from raw)
        │
        └─── Segmentation input (all methods) ──► Detected masks
                    (raw images only)                    │
                                                         ▼
                                              Quantitative measurement
                                              (raw images × masks)
                                                         │
                                                         ▼
                                                   CSV output
```

---

## 4. Per-Method Segmentation Rules

### 4.1 SAM (Segment Anything)

- **Display for annotation**: processed image shown in canvas. User draws box or point prompts on processed image.
- **SAM input**: raw aligned image loaded from `preprocessing_cache[position_key]['raw'][channel_name]`.
- **Coordinate mapping**: prompts drawn on the processed image are valid for the raw image without transformation, because preprocessing does not spatially resample.
- **Why raw for SAM**: SAM's encoder responds to edge contrast and texture that must reflect actual signal boundaries, not contrast-enhancement artifacts. Using a CLAHE-processed image would cause SAM to segment contrast-enhanced noise as signal.

### 4.2 Blob Detection (threshold + morphology)

- **Display**: processed image for visual reference.
- **Blob input**: raw image only. No fallback to processed.
- **Threshold parameter**: the slider value (0–1) is applied to the raw image's normalized intensity range. This ensures the threshold represents a meaningful percentile of the raw signal distribution.
- **Why raw**: background subtraction and CLAHE in preprocessing alter local mean and variance in ways that make the threshold value non-reproducible across images. Raw-based thresholds are reproducible.

### 4.3 Random Forest (pixel-level classifier)

- **Display for annotation**: processed image shown in canvas. User paints exosome and background strokes on the processed image.
- **Coordinate mapping**: annotation pixel coordinates (x, y) are the same in both processed and raw images. No remapping needed.
- **Feature extraction**: `extract_features()` receives the raw image array. The 8 features (raw intensity, Gaussian×3, LoG, gradient magnitude, local variance) are therefore derived from unmodified raw signal values.
- **Training data**: the label assigned to pixel (x, y) is whatever the user painted, but the feature vector at that pixel is read from the raw image at (x, y).
- **Prediction**: full-image prediction also runs on raw features.
- **Why raw**: RF learns the relationship between raw pixel texture and the biological signal. If trained on processed features, the classifier would fail on any image with different preprocessing settings.

---

## 5. Annotation Coordinate Mapping

Preprocessing operations applied by `preprocess_engine.py`:
- Denoising (NLM): intensity-only, no spatial change
- Background subtraction (morphological opening): intensity-only
- CLAHE: intensity-only
- Gamma correction: intensity-only
- Linear contrast/brightness: intensity-only
- Unsharp masking: intensity-only
- Thresholding: intensity-only

**None of these operations change image dimensions or pixel coordinates.** A pixel annotated at (x=150, y=200) on the processed display image is exactly pixel (150, 200) in the corresponding raw image.

This means the frontend annotation system requires no coordinate transform. The annotations collected from the canvas are directly usable as (x, y) indices into the raw numpy array.

The only case requiring geometric correction would be if the display canvas is rendered at a different zoom/scale than the image resolution. The canvas must therefore always operate at 1:1 pixel correspondence with the source image (either through CSS `image-rendering: pixelated` at 100% zoom or by passing a `scale_factor` from the frontend that is used to divide annotation coordinates before use).

---

## 6. Quantitative Output Requirements

After segmentation by any method, the detected masks define object regions. All reported values must be extracted from raw images using those masks.

### 6.1 Shape measurements (from mask geometry — method-independent)

These are computed from the binary mask alone and do not depend on image intensity:

| Field | Computation |
|-------|-------------|
| `area` | `np.sum(mask)` — pixel count |
| `centroid_x`, `centroid_y` | Image moments |
| `bbox_x1/y1/x2/y2` | Bounding box of mask |
| `perimeter` | `cv2.arcLength` on mask contour |
| `circularity` | `4π·area / perimeter²`, clamped to [0, 1] |

These are already implemented. They are method-independent and correct regardless of which image was used for segmentation, because they are derived from the mask shape, not pixel values.

### 6.2 Intensity measurements (from raw image × mask)

These must be added and must use raw images:

| Field | Computation | Source |
|-------|-------------|--------|
| `mean_intensity_<channel>` | `np.mean(raw[mask])` | Raw image for each aligned channel |
| `max_intensity_<channel>` | `np.max(raw[mask])` | Raw image |
| `integrated_intensity_<channel>` | `np.sum(raw[mask])` | Raw image |
| `snr_<channel>` | `mean_intensity / background_std` | Raw image; background = inverted mask region |

### 6.3 Colocalization / spatial relation counts

- Computed from the detected masks of different channels intersected with each other.
- Intersection logic operates on mask arrays (geometry), not intensities.
- Reported counts are therefore raw-independent, but confirmation intensities (e.g., mean intensity of a co-localizing object in a secondary channel) must come from the raw secondary channel image.

### 6.4 CSV schema (target)

```
id, area, perimeter, circularity,
centroid_x, centroid_y,
bbox_x1, bbox_y1, bbox_x2, bbox_y2,
score,
mean_intensity_ch1, mean_intensity_ch2, ..., mean_intensity_chN,
max_intensity_ch1, ..., max_intensity_chN,
integrated_intensity_ch1, ..., integrated_intensity_chN,
snr_ch1, ..., snr_chN,
colocalized_with_ch2, colocalized_with_ch3, ...
```

All `*_intensity_*` and `snr_*` columns are derived from raw aligned images. Shape columns (area, perimeter, circularity, bbox, centroid) are derived from mask geometry.

---

## 7. Implementation Tasks (ordered)

### Phase 1 — Enforce raw-only segmentation inputs

**File: `api_server_extended.py`, function `exosome_segment()`**

Replace the current image resolution logic (which prefers processed stages) with a strict policy:

```python
# ALWAYS use raw image for segmentation input
image_path = None
position_key = f"{sample_name}/{position_name}"

if position_key in preprocessing_cache:
    raw_stage = preprocessing_cache[position_key].get('raw', {})
    if channel_name in raw_stage:
        image_path = raw_stage[channel_name]

if not image_path or not image_path.exists():
    return jsonify({'success': False,
                    'error': f'Raw image not found for {sample_name}/{position_name}/{channel_name}. '
                             f'Load the sample first.'}), 404
```

Remove all fallbacks to processed stages for segmentation input. The processed image remains accessible separately for display purposes via the existing `/api/input/preprocess/final` and `/api/images/` endpoints.

### Phase 2 — Enforce raw features in RF

**File: `api_server_extended.py`, function `exosome_segment()`, RF branch**

The `image_array` passed to `segment_with_random_forest()` must come from the raw path resolved above. After Phase 1 this is automatically satisfied because `image_array` is always loaded from raw. No changes to `random_forest_segmentation.py` are needed; the enforcement is at the call site.

Add a log line to make this explicit and auditable:
```python
print(f"[RF] Feature extraction from raw image: {image_path}, dtype={image_array.dtype}")
```

### Phase 3 — Display track: load processed image separately for the canvas

**File: `api_server_extended.py` (or a new endpoint)**

The frontend canvas must show the processed image, not the raw one. This separation must be explicit. Add a dedicated endpoint or document that the existing `/api/input/preprocess/final` endpoint is the display source. The frontend should:

1. Request the processed image for display via the existing preview endpoint.
2. Send annotations (pixel coordinates) to the segmentation endpoint.
3. Never assume the displayed image is the one used for segmentation.

No backend changes needed here beyond clear documentation in the API docstring.

### Phase 4 — Raw-based intensity measurement after segmentation

**New function: `exosome_detection/intensity_extractor.py`**

```python
def extract_raw_intensities(
    masks: list[np.ndarray],          # List of (H, W) bool arrays
    raw_channels: dict[str, np.ndarray],  # {channel_name: raw_array}
) -> list[dict]:
    """
    For each mask, measure mean/max/integrated intensity
    from each raw channel. Background estimated from inverted mask.
    Returns list of dicts, one per mask.
    """
```

Called inside `exosome_segment()` after detection, using the raw image(s) in `preprocessing_cache[position_key]['raw']` for all available channels.

### Phase 5 — CSV export with raw measurements

**File: `api_server_extended.py`, function `exosome_export()`**

Extend the CSV writer to include:
- `perimeter` and `circularity` (already in `detections` dict from Phase 0 work)
- `mean_intensity_<ch>`, `max_intensity_<ch>`, `integrated_intensity_<ch>` for all available raw channels
- Add a `image_source: raw` field to `run.json` to explicitly record that measurements came from the raw image

### Phase 6 — Frontend: visually distinguish display image from analysis image

**File: `frontend/src/components/ExosomeDetection.tsx`**

- Label the image canvas clearly: "Visualization (preprocessed)" vs. the analysis source.
- When the user switches preprocessing settings, show a warning if annotations exist: "Annotations were drawn on a different preprocessing. The annotation coordinates are still valid (raw-based analysis is unaffected), but the visual reference has changed."
- Add a toggle to show either the processed image or the raw image in the canvas (for expert review), while keeping the annotation layer independent.

---

## 8. Invariants to Enforce (non-negotiable)

1. `segment_exosomes()`, `detect_blobs()`, and `segment_with_random_forest()` must never receive a processed image as input in production code paths.
2. `extract_features()` in `random_forest_segmentation.py` must receive the raw array. No caller may pass a preprocessed array.
3. The CSV `mean_intensity_*` columns must be populated from raw arrays. If raw arrays are unavailable, the column must be written as empty/null, never from the processed array.
4. The preprocessing pipeline may change at any time (user can adjust CLAHE, gamma, etc.) without invalidating previously computed shape measurements (area, perimeter, circularity), because those come from the mask, not pixel values.
5. All of the above must hold regardless of whether the raw image is 8-bit, 16-bit, or float. The raw loader must preserve original bit-depth; normalization to [0,1] for RF features is applied internally, but the raw array is never permanently converted to 8-bit for analysis purposes.

---

## 9. What Does NOT Change

- The preprocessing pipeline itself is unchanged and correct for its display purpose.
- Annotation UX (painting strokes, drawing boxes) is unchanged.
- Detection algorithm implementations (`sam_service.py`, `blob_service.py`, `random_forest_segmentation.py`) are unchanged internally.
- The perimeter and circularity additions from the previous implementation cycle are correct and remain.
- Shape measurements (area, perimeter, circularity, bbox, centroid) are already raw-independent and correct.

---

## 10. Summary of Role Boundaries

```
Processed image
  ├── Shown in display canvas                         YES
  ├── Used for user annotation guidance               YES
  ├── Input to SAM segmentation                        NO  (use raw)
  ├── Input to blob threshold                          NO  (use raw)
  ├── Input to RF feature extraction                   NO  (use raw)
  ├── Source of mean/max intensity in CSV              NO  (use raw)
  └── Source of threshold parameter calibration        NO  (use raw)

Raw image
  ├── Shown in display canvas (optional expert mode)  YES
  ├── Input to all segmentation methods               YES
  ├── Input to RF feature extraction                  YES
  ├── Source of all intensity measurements in CSV     YES
  ├── Source of SNR calculation                       YES
  └── Source of colocalization intensity confirmation YES
```
