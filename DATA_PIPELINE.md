# SEA Data Pipeline Reference

This document maps file/data flow across the main user workflow stages:

1. Data loading
2. Alignment
3. Exosome detection / segmentation
4. Results export / viewing
5. Data analysis

Use this as a lookup for **what creates each file**, **where it is saved**, and **which stage consumes it**.

---

## Path Conventions

- `DATA_ROOT` = app input root (in current backend config this is `<data_root>/input`)
- Paths below are shown **relative to `DATA_ROOT`** unless explicitly marked “outside DATA_ROOT”.
- `<sample>` and `<position>` mean the currently selected sample/position IDs.
- `<channel>` means channel key like `C1_ch1`, `C0`, etc.

---

## 1) Data Loading Stage (`/api/input/load_position`)

### Inputs read

| Input | Relative path from `DATA_ROOT` | Required |
|---|---|---|
| Raw channel TIFFs for selected position | `<sample>/<position>/*.tif*` | Yes |
| Saved preprocess state (sample-level) | `<sample>/.sea_state/preprocess_state.json` | Optional |
| Marker mapping workbook | Outside `DATA_ROOT` (`data/label/Marker_info.xlsx`) | Optional |

### Outputs generated

| Output | Saved path | Notes |
|---|---|---|
| Position/channel metadata returned to frontend (`items`, previews, stage info) | API response only (in memory) | Includes per-channel keys and preview URLs |
| In-memory preprocessing cache seed (`raw`) | In memory (`preprocessing_cache["<sample>/<position>"]["raw"]`) | Used by preprocessing/alignment/detection |
| Aligned-artifact availability flags | API response fields: `align_artifact_channels`, `has_align_artifacts` | Derived from `<sample>/align/<position>/*_aligned.tif` |

### Consumed by next stages

- **Alignment** consumes `raw` cache entries and optional preprocessed stage state.
- **Exosome detection** uses loaded channel list and selected channel.
- **UI** uses `align_artifact_channels` to show “Using aligned image”.

---

## 2) Alignment Stage (`/api/input/align`)

### Inputs read

| Input | Relative path from `DATA_ROOT` | Required |
|---|---|---|
| Position existence check | `<sample>/<position>/` | Yes |
| Raw / preprocessed channel images (via cache restored from disk) | Originally from `<sample>/<position>/*.tif*`, plus possible `.sea_state` referenced outputs | Yes |
| Alignment settings state (optional restore) | `<sample>/.sea_state/alignment_state.json` | Optional |
| Manual ROI alignment inputs | API request body (`diagonal_boxes`, etc.) | Optional (manual mode) |

### Outputs generated

| Output | Saved path | Notes |
|---|---|---|
| Per-channel aligned TIFF mirror | `<sample>/align/<position>/<channel>_aligned.tif` | **New auto-copy mirror** for downstream segmentation priority |
| Alignment state snapshot | `<sample>/.sea_state/alignment_state.json` | Written when user clicks Save in Alignment UI |
| Alignment previews + transform stats | API response / preview cache | For UI visualization |
| Canonical aligned stage files | Outside `DATA_ROOT` (`PROCESSING_OUTPUT/<sample>/<position>/aligned_from_<input_stage>/<channel>_aligned.tif`) | Existing behavior unchanged |

### Consumed by next stages

- **Exosome detection resolver** now checks aligned mirror first:
  - `<sample>/align/<position>/<channel>_aligned.tif`
- If not found, it falls back to preexisting raw/processed resolution logic.

---

## 3) Exosome Detection / Segmentation Stage (`/api/exosome/segment`, `/api/exosome/auto_segment`)

### Inputs read

| Input | Relative path from `DATA_ROOT` | Required |
|---|---|---|
| Preferred segmentation source (new priority #1) | `<sample>/align/<position>/<channel>_aligned.tif` | Optional (if present, used first) |
| Fallback raw source | `<sample>/<position>/*.tif*` | Used if aligned mirror missing |
| Optional fallback processed source | Restored cache references from preprocess/session state | Used when raw channel not available in cache |
| Prompt/annotation payload | API request body | Depends on detection mode |

### Outputs generated

| Output | Saved path | Notes |
|---|---|---|
| Detection overlay image | Outside `DATA_ROOT`: `OUTPUT_ROOT/exosome_detection/<sample>/<position>/<channel>/<channel>_overlay_<timestamp>.png` | Persisted detection result |
| Detection masks archive | Outside `DATA_ROOT`: `.../<channel>_masks_<timestamp>.npz` | Compressed masks + scores |
| Detection table | Outside `DATA_ROOT`: `.../<channel>_results_<timestamp>.csv` | area/perimeter/circularity/centroid/bbox/score |
| Run metadata | Outside `DATA_ROOT`: `.../<channel>_run_<timestamp>.json` | method/settings/time info |

### Consumed by next stages

- **Results Viewer** loads these files from `OUTPUT_ROOT/exosome_detection/...`.
- **Colocalization analysis** reads latest per-channel detection CSV/mask artifacts.

---

## 4) Results Export / Viewing Stage (Results Viewer + Colocalization)

This stage has two data families:

1. Detection artifacts (outside `DATA_ROOT`, under `OUTPUT_ROOT/exosome_detection/...`)
2. Colocalization outputs (inside `DATA_ROOT`, under `<sample>/results/<position>/`)

### Inputs read

| Input | Relative path from `DATA_ROOT` | Required |
|---|---|---|
| Colocalization persisted outputs (preferred) | `<sample>/results/<position>/colocalization_*.{json,csv,png}` | Optional (if already generated) |
| Reference table for Excel export | `<sample>/results/<position>/colocalization_reference_table.{json,csv}` | Optional |
| Detection exports for current sample/position | Outside `DATA_ROOT`: `OUTPUT_ROOT/exosome_detection/<sample>/<position>/<channel>/...` | Required for full results view |

### Outputs generated

| Output | Saved path | Notes |
|---|---|---|
| Colocalization reference CSV | `<sample>/results/<position>/colocalization_reference_table.csv` | Saved by `/api/colocalization_analysis` |
| Colocalization reference JSON | `<sample>/results/<position>/colocalization_reference_table.json` | Same run |
| Colocalization summary JSON | `<sample>/results/<position>/colocalization_summary.json` | Same run |
| Combination summary CSV | `<sample>/results/<position>/colocalization_combinations.csv` | Same run |
| Positive overlay PNG | `<sample>/results/<position>/colocalization_positive_overlay.png` | Optional if overlay generation succeeds |

### Consumed by next stages

- **Results Viewer** displays tables/summary from these saved colocalization files.
- **Excel export endpoint** (`/api/exosome/export_excel`) reads the reference table from `<sample>/results/<position>/...`.

---

## 5) Data Analysis Stage (`/api/cygnus/run`)

### Inputs read

| Input | Relative path from `DATA_ROOT` | Required |
|---|---|---|
| Uploaded analysis tables (`.csv`, `.xlsx`, `.xls`) | Not read from `DATA_ROOT`; uploaded via multipart form (`files[]` / `file`) | Yes |
| Validation schema for fixed/marker columns | In code (`cygnus_upload_merge.py`, `stage1_loader.py`) | Yes |

### Processing/output behavior

| Output | Saved path | Notes |
|---|---|---|
| Merged upload table | Temporary file outside `DATA_ROOT` | Created by server (`pd.concat(..., ignore_index=True)`) |
| Pipeline output directory | Temporary directory outside `DATA_ROOT` | `run_full_pipeline(..., output_dir=tempdir)` |
| Rendered report HTML | Returned in API response (`report_html`) | Not persisted under `DATA_ROOT` by this endpoint |

### Consumed by next stages

- **Frontend Data Analysis tab** renders returned HTML directly.
- No downstream filesystem stage depends on these temporary files (they are cleaned up).

---

## Quick “File Origin” Lookup

| File/folder | Created by |
|---|---|
| `<sample>/<position>/*.tif*` | User data (input) |
| `<sample>/align/<position>/<channel>_aligned.tif` | Alignment stage (`/api/input/align`) mirror copy |
| `<sample>/results/<position>/colocalization_*` | Results Viewer colocalization run (`/api/colocalization_analysis`) |
| `<sample>/.sea_state/alignment_state.json` | Alignment Save action |
| `<sample>/.sea_state/preprocess_state.json` | Preprocess state persistence |
| `OUTPUT_ROOT/exosome_detection/<sample>/<position>/<channel>/*` | Exosome segmentation persistence |

