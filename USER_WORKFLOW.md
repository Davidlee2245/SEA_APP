# SEA — User-Facing Workflow Documentation

**SEA (Semiautomated Exosome Analysis)** is a desktop/web application for analyzing multi-channel fluorescence microscopy images. It detects, quantifies, and colocates exosomes across imaging channels.

There are two ways to use SEA:

- **GUI** — a five-tab browser/Electron interface at `http://localhost:3000`
- **CLI** — a batch command-line pipeline for automated runs

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [GUI Overview — Five Tabs](#2-gui-overview--five-tabs)
3. [Tab 1: Agent Chat](#3-tab-1-agent-chat)
4. [Tab 2: Image Processing](#4-tab-2-image-processing)
5. [Tab 3: Alignment](#5-tab-3-alignment)
6. [Tab 4: Exosome Detection](#6-tab-4-exosome-detection)
7. [Tab 5: Results Viewer](#7-tab-5-results-viewer)
8. [CLI Pipeline (Batch Mode)](#8-cli-pipeline-batch-mode)
9. [Data Input / Output Reference](#9-data-input--output-reference)

---

## 1. Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support
- Conda environment (`SEA`) installed per the README
- OpenAI API key (optional — required only for Agent Chat and LLM analysis)

### Starting the Application

**Step 1 — Start the backend (Terminal 1):**

```bash
conda activate SEA
python api_server_extended.py
# Backend is ready at http://localhost:5000
```

**Step 2 — Start the frontend (Terminal 2):**

```bash
cd frontend
npm run dev
# Open http://localhost:3000 in your browser
```

**Step 3 — Open your browser** and navigate to `http://localhost:3000`. You will see the five-tab navigation bar at the top of the page.

### Input Data Structure

Before using the application, organize your microscopy data as follows:

```
data/input/
├── Sample_01/
│   ├── ch1_CD63.tif        ← 16-bit or 32-bit grayscale TIFF
│   ├── ch2_CD81.tif
│   └── ch3_Syntenin.tif
└── Sample_02/
    └── ...
```

- Each sample must be in its own subdirectory
- Files must be `.tif` or `.tiff` format

---

## 2. GUI Overview — Five Tabs

The application is organized into five tabs, accessed via the navigation bar at the top of every page. Tabs preserve their state when you switch between them — you do not lose your work.

| Tab | Purpose |
|-----|---------|
| 🤖 Agent Chat | AI-powered preprocessing advisor — upload images, ask questions, get parameter recommendations |
| 🖼️ Image Processing | Load channels and apply preprocessing steps (contrast, background subtraction, blur) |
| 🎯 Alignment | Register channels to a common reference frame using automated or manual methods |
| 🔬 Exosome Detection | Run exosome segmentation on aligned channels; draw prompts, annotate, and export |
| 📊 Results Viewer | View final outputs, overlay images, CSV data, and colocalization statistics |

The recommended workflow flows left to right through the tabs: **Preprocess → Align → Detect → View Results**.

---

## 3. Tab 1: Agent Chat

### Purpose

An AI-powered chat assistant (backed by OpenAI GPT-4o-mini) that analyzes your microscopy images and recommends optimal preprocessing parameters.

### Entry Point

Click **🤖 Agent Chat** in the navigation bar.

### Workflow

#### 3a. Check API Status

- On load, the header shows either **✅ OpenAI API Ready** or **⚠️ API Not Configured**.
- If not configured: follow the on-screen instructions to set `OPENAI_API_KEY` and start `api_agent_chat.py` on port 5001, then refresh.

#### 3b. Upload an Image (Optional but Recommended)

1. Click **🖼️ Upload Image** in the input area at the bottom.
2. Select one or more image files (TIFF, PNG, JPEG, etc., up to 50 MB each).
   - **TIFF files** are automatically converted to PNG previews for display.
3. Thumbnails of the uploaded images appear above the text input.
4. Click any thumbnail to open a full-screen preview modal.
5. To remove an image: click the **✕** button on its thumbnail, or click **Clear All**.

#### 3c. Ask a Question or Request a Recommendation

**Option A — Free-form conversation:**
- Type any question in the text box (e.g., *"My images have high background noise — what do you recommend?"*)
- Press **Enter** or click **📤 Send**
- The assistant replies with analysis and parameter suggestions
- Use **Shift+Enter** to add a new line without sending

**Option B — Quick Recommend (one click):**
- Only available after uploading an image
- Click **⚡ Quick Recommend** in the header
- The assistant automatically analyzes the image and returns structured preprocessing parameters (contrast, denoising, background subtraction settings)

#### 3d. Review and Act on Recommendations

- Assistant replies appear in the chat thread with a timestamp
- If the reply includes detected parameters, a **📊 Recommended Parameters** block appears below the message
- From that block, you can:
  - **📋 Copy Parameters** — copies the JSON to your clipboard
  - **✅ Approve** — saves the parameters to the manifest and creates a human-readable `.txt` file in `data/input/`
  - **✏️ Edit** — opens an inline JSON editor to modify values before saving

#### 3e. Decision Points

| Situation | What Happens |
|-----------|-------------|
| Image not uploaded and Quick Recommend clicked | A warning message appears in chat; no API call is made |
| API key not set | All send buttons are disabled |
| Network error or server down | Error message appears in chat |

#### 3f. Clear Chat History

Click **🗑️ Clear Chat** and confirm the dialog to erase all messages and start fresh.

**Outcome:** A set of recommended preprocessing parameters saved to `data/input/` or copied to clipboard, ready to use in the Image Processing tab.

---

## 4. Tab 2: Image Processing

### Purpose

Load your sample and imaging position, preview raw channel images, and apply preprocessing steps (contrast enhancement, background subtraction, Gaussian blur) before alignment.

### Entry Point

Click **🖼️ Image Processing** in the navigation bar.

### Layout

The tab uses a three-column layout:

- **Left sidebar** — File selection controls and processing step options
- **Center panel** — Image preview for the selected channel
- **Right panel** — Channel statistics and histogram

### Workflow

#### 4a. Select and Load a Sample

1. In the **File Operations** section of the left sidebar, choose a **Sample** from the dropdown (lists all subdirectories found in `data/input/`).
2. Choose a **Position** from the second dropdown (e.g., P1, P2).
3. Click **Load Position**.
4. The center panel shows a preview of the first channel; a channel selector appears.

> If the default sample (`A2780Cis10`, position `P1`) is present in `data/input/`, it loads automatically on page open.

#### 4b. Browse Channels

- Use the **Channel** dropdown in the center panel to switch between available imaging channels (e.g., `C1_ch1(p62)`, `C2_ch2(CD63)`).
- If preprocessing was done previously, the processed (final) result is shown automatically; otherwise the raw image is shown.

#### 4c. Apply Contrast Enhancement

1. In the left sidebar under **⚙️ Image Processing → Contrast Enhancement**, choose a method:
   - **CLAHE** — adjust **Clip Limit** (1.0–6.0) and **Tile Grid Size** (4–16) using sliders
   - **Linear Stretch** — adjust **Low Percentile** (0–10%) and **High Percentile** (90–100%)
2. Choose scope:
   - Check **Apply to All Channels** to process every channel at once
   - Leave unchecked to process only the currently selected channel
3. Click **Apply to All Channels** (or **Apply to [Channel Name]**).
4. The center preview updates to show the contrast-enhanced result.

#### 4d. Apply Background Subtraction

1. Select the target channel using the center panel dropdown.
2. In the left sidebar under **Background Subtraction**, set the **Strength** (0–200).
3. Click **Apply to [Channel Name]**.
4. Preview updates to show background-subtracted result.

#### 4e. Apply Gaussian Blur

1. Select the target channel.
2. In the left sidebar under **Gaussian Blur**, set **Sigma** (0.1–10.0).
3. Click **Apply to [Channel Name]**.
4. Preview updates.

#### 4f. Current Stage Indicator

The sidebar shows a **Current output stage** label that tracks the most recent processing step applied (Raw → Contrast Enhanced → Bkg Subtracted → Gaussian Blurred).

#### 4g. Reset to Original

Click **Reset to Original** to discard preprocessing for the selected channel and revert to the raw image. Other channels are unaffected.

#### 4h. Session Persistence

Preprocessing steps are saved automatically to a server-side session. When you reload the position, the previously applied steps are re-applied in order.

**Outcome:** Each channel has a final processed image ready to hand off to the Alignment tab. The processed images are stored on disk and automatically picked up when you switch to the Alignment tab.

---

## 5. Tab 3: Alignment

### Purpose

Register all channels to a common reference frame, correcting for physical offsets between imaging cycles. The workflow has two stages: feature detection (preview), then alignment execution.

### Entry Point

Click **🎯 Alignment** in the navigation bar.

### Workflow

#### 5a. Select and Load a Sample

1. Choose a **Sample** and **Position** in the left sidebar (same selectors as Image Processing).
2. Click **Load Position**.
3. Processed previews from the Image Processing tab are automatically fetched and displayed.

> **Note:** When you switch to this tab while Image Processing is active, processed previews are refreshed automatically without requiring a reload.

#### 5b. Configure Channels and Reference

- **Channel checkboxes** — by default all channels are selected for alignment. Uncheck any channel to exclude it.
- **Reference Channel** — select the channel that all others will be registered to (the anchor). The reference channel is never warped.
- **Input Stage** — select which preprocessing stage to use as the alignment input (e.g., Contrast Enhanced, Gaussian Blurred). Defaults to the most advanced stage available.

#### 5c. Choose an Alignment Method

Select a method from the **Alignment Method** dropdown:

| Method | Best For | What It Does |
|--------|---------|--------------|
| **Frequency Domain FFT** | Translational / small-rotation offsets | Detects grid spacing from Fourier spectrum; estimates shift via phase correlation |
| **Grid Intersection (Homography)** | Regular grid patterns | Finds corner intersections and computes a projective transform |
| **Grid Line (Hough)** | Grids with clear line features | Detects horizontal/vertical lines, finds intersections |
| **Manual Diagonal** | Fine manual control | User draws diagonal bounding boxes on each channel; offset is computed from box positions |

Each method exposes its own tuning parameters in the left sidebar (e.g., FFT: window function, spacing range; Grid: blur kernel, threshold method, Hough thresholds).

#### 5d. Stage A — Detect Features (Preview)

> *Skip this stage if using Manual Diagonal method.*

1. Click **Detect Features**.
2. The system processes each selected channel and overlays detected features (corners, grid lines, or FFT spectrum) on the preview.
3. A **feature summary** text appears below (e.g., number of corners, grid spacing).
4. Use the **Preview Layer** dropdown to switch between views:
   - `preprocessed`, `binary_mask`, `detected`, `intersections` (grid methods)
   - `fft_spectrum`, `synthetic_grid` (FFT method)
5. If detection looks poor, adjust the method parameters and click **Detect Features** again.

#### 5e. Stage A (Manual Diagonal Only) — Draw Bounding Boxes

1. For each channel, a canvas is shown with the preprocessed image.
2. Draw a diagonal bounding box around the region of interest.
3. The reference channel box defines the "ideal" position; target channel boxes define their current position.
4. Adjust the box position, size, and rotation by dragging handles.

#### 5f. Stage B — Run Alignment

1. When Stage A is complete (or boxes are drawn for Manual Diagonal), click **Run Alignment**.
2. The backend computes affine transforms and warps each non-reference channel onto the reference.
3. After completion:
   - The center preview automatically switches to show the aligned result for the first non-reference channel.
   - An **Alignment Shift Panel** on the right shows dx, dy, rotation, residual error, and number of matched features for each channel.
   - The stage indicator advances to `Aligned`.
4. Use the **Channel** and **Stage** selectors to compare raw, processed, and aligned previews side by side.

#### 5g. Color Overlay

- A **Color Overlay** panel lets you toggle visibility of each channel and view a composite RGB overlay of the aligned result.
- Click any channel name in the shift panel to highlight it in the overlay.

#### 5h. Save / Restore Alignment

- Click **Save** to persist the current alignment configuration (method, shift vectors, manual boxes) to browser local storage for this sample/position.
- Click **Load Saved** to restore a previously saved configuration.
- Click **Reset** to clear all saved data and start over.

**Outcome:** All channels are aligned to a shared coordinate space. Aligned images are saved to `data/` and are automatically available for Exosome Detection.

---

## 6. Tab 4: Exosome Detection

### Purpose

Segment and count exosomes within each aligned channel image using one of three detection engines. Supports interactive point/box prompting, brush annotation, and result export.

### Entry Point

Click **🔬 Exosome Detection** in the navigation bar.

### Workflow

#### 6a. Select and Load a Sample

1. Choose a **Sample** and **Position** in the left sidebar.
2. Click **Load Position**.
3. The aligned channel image loads onto an interactive canvas. Ground-truth points (if a CSV file exists) are auto-loaded.

#### 6b. Select a Channel

- Use the **Channel** dropdown to switch between available channels.
- Click history and annotations auto-save per channel and are restored when you return.

#### 6c. Adjust Display Mode

Use the display controls to adjust how the image looks (does not affect detection):

- **Display Mode**: `raw_16bit`, `enhanced`, `minmax`
- **LUT**: `gray` or `red`
- Normalization statistics (min, max, percentiles) are shown in a panel on the right.

#### 6d. Choose a Detection Method

Select a method in the left sidebar:

| Method | How to Use | Best For |
|--------|------------|---------|
| **SAM** (Segment Anything Model) | Click-based or box prompts | General-purpose; works on varied shapes |
| **Blob** | Automatic; configure thresholds | Round, well-separated particles |
| **Random Forest** | Draw positive/negative annotations with a brush | When SAM/Blob give too many false positives |

#### 6e. Option A — SAM Detection

1. Set **Detection Mode**:
   - **Auto** — no user input required; SAM segments the entire image
   - **Box** — click and drag on the canvas to draw a bounding box around the region of interest
   - **Point** — click on individual exosomes to add positive prompts; right-click to add negative prompts
2. Set the **Confidence Threshold**, **Min Area**, and **Max Area** sliders to filter detections.
3. Toggle **Remove Small Objects** and **Fill Holes** as needed.
4. Click **Run Detection**.

#### 6f. Option B — Blob Detection

1. Select **Blob** as the detection method.
2. Adjust parameters:
   - **Threshold** — pixel intensity cutoff for foreground
   - **Min / Max Circularity** — shape filter (0 = any shape, 1 = perfect circle)
   - **Min Inertia Ratio** — elongation filter
3. Click **Run Detection**.

#### 6g. Option C — Random Forest Detection

1. Select **Random Forest** as the detection method.
2. Choose **Annotation Mode**: `Exosome` (positive) or `Background` (negative).
3. Set **Brush Size**.
4. Paint on the canvas: left-click drag over exosomes to label them as positive; switch mode and paint background regions as negative.
5. Click **Run Detection** to train and infer in one step.
6. The **Probability Map** toggle shows a heatmap of confidence values.

#### 6h. Review Detections on Canvas

After detection runs:
- Detected objects are overlaid on the canvas (outlines and/or filled masks).
- An **Area Distribution Histogram** appears below the canvas, showing the size distribution of detections.
- A **detections count** badge shows the total number found.
- Toggle **Show Mask Outlines** / **Mask Opacity** to adjust overlay visibility.
- Click any detection on the canvas to see its properties (area, circularity, intensity, confidence).

#### 6i. Ground Truth Comparison

If a ground truth CSV file is found for the sample/position, ground truth dot positions are drawn on the canvas (toggleable via **Show Ground Truth**).

#### 6j. Click History and Session Persistence

- All prompts (box draws, point clicks, annotations) are saved to browser local storage automatically.
- When you return to the same channel and method, the previous prompts are restored.
- Click **Save** / **Load** / **Reset** in the sidebar to manage click history explicitly.

#### 6k. Export Results

1. Click **Export Detections** (or similar export button in the sidebar).
2. The system generates:
   - A CSV file with object coordinates, area, circularity, intensity, and confidence for each detected exosome
   - Label images (.tif) for each channel
3. Export status and file paths are shown in the sidebar.

**Outcome:** Per-channel exosome segmentation results saved as CSV and label images, available in the Results Viewer tab.

---

## 7. Tab 5: Results Viewer

### Purpose

Load and browse the final outputs from all previous stages: overlay images, detection CSV files, label images, and colocalization statistics.

### Entry Point

Click **📊 Results Viewer** in the navigation bar.

### Workflow

#### 7a. Load a Sample's Results

1. Select a **Sample** and **Position** from the dropdowns.
2. Click **Load Position**.
3. The viewer fetches exosome detection results for that position. If none are found, it falls back to alignment-only results.

> The default sample (`A2780Cis10`, position `P1`) loads automatically if present.

#### 7b. Browse Output Files

The **Results** panel lists all output files for the loaded position, categorized with icons:

| Icon | Type | Content |
|------|------|---------|
| 🎨 | Overlay | RGB composite image of all aligned channels |
| 📊 | CSV | Detections table (x, y, area, intensity, circularity, confidence) |
| 📈 | Plot | Statistical plots generated by LLM analysis (if enabled) |
| 🏷️ | Label | Per-channel label images with segmented objects |
| ✓ | Registered | Aligned channel images |

- Click any file in the list to preview it in the center panel (images are displayed inline; CSV and JSON are described).
- Click the **⬇️ Download** button next to any file to save it locally.

#### 7c. Colocalization Analysis

After loading results, a **Colocalization** section appears below the file list:

1. Select the **Reference Channel** (the channel to compare all others against).
2. Select one or more **Marker Channels** to analyze.
3. Choose an **Analysis Mode**:
   - **Nearest Centroid** — reports distance from each detected exosome to the nearest detection in the reference channel
   - **Overlap** — reports pixel-level overlap between channel masks
4. Set the **Distance Threshold** (pixels) for colocalization calls.
5. Click **Run Colocalization**.
6. Results appear as a table showing, for each channel pair:
   - Total detections in each channel
   - Number and percentage colocalized
   - Mean / median distance between colocalized pairs

**Outcome:** A complete view of all analysis outputs with downloadable files and interactive colocalization statistics.

---

## 8. CLI Pipeline (Batch Mode)

### Purpose

Process multiple samples automatically without the GUI, using a three-phase pipeline: Inspector → Aligner → Analyst.

### Entry Point

```bash
conda activate SEA
python pipeline.py
```

### Pipeline Phases

| Phase | Agent | What It Does |
|-------|-------|-------------|
| 1 — Inspector | `Inspector` | Calculates SNR for each channel; applies denoising (Noise2Void or rule-based); identifies the sharpest (anchor) channel |
| 2 — Aligner | `Aligner` | Registers all channels to the anchor using SuperPoint + SuperGlue feature matching and affine/TPS warping |
| 3 — Analyst | `Analyst` | Detects exosomes with StarDist; performs colocalization analysis; exports CSV and overlay images |

### Commands

```bash
# Process all samples in data/input/
python pipeline.py

# Process specific samples only
python pipeline.py --samples Sample_01 Sample_02

# Use a custom input directory
python pipeline.py --input /path/to/your/data

# Use a custom config file
python pipeline.py --config config/custom_config.yaml
```

### Decision Points

| Situation | Behavior |
|-----------|----------|
| No samples found in input directory | Pipeline exits with an error message |
| A phase fails for a sample | That sample is marked as failed; remaining samples continue |
| CUDA not available | Falls back to CPU automatically (with a warning) |
| LLM API key not set | LLM quality assessment and statistical interpretation are skipped; pipeline completes with rule-based decisions only |

### Output

For each processed sample, the pipeline creates:

```
data/output/
└── Sample_01/
    ├── preprocessed/
    │   └── ch1_CD63_preprocessed.tif      ← Denoised images
    ├── registered/
    │   └── ch1_CD63_registered.tif        ← Aligned images
    ├── labels/
    │   └── ch1_CD63_labels.tif            ← Segmentation label images
    ├── Sample_01_overlay.png              ← RGB composite overlay
    └── Sample_01_detections.csv          ← Per-exosome data table
```

### Summary Report

After all samples complete, the terminal prints:

```
Pipeline Summary
Total samples:  N
Successful:     M
Failed:         K
```

The process exits with code `0` if all samples succeeded, or `1` if any failed.

---

## 9. Data Input / Output Reference

### Input File Requirements

- Format: `.tif` or `.tiff`
- Bit depth: 16-bit or 32-bit grayscale
- Structure: one file per channel, one directory per sample
- Naming: any naming convention (detected by file extension)

### CSV Output Format

The `*_detections.csv` file produced by the Analyst (CLI) or exported from Exosome Detection (GUI) contains:

| Column | Description |
|--------|-------------|
| `channel` | Channel name |
| `object_id` | Unique detection identifier |
| `x_coord`, `y_coord` | Pixel coordinates of centroid |
| `intensity` | Normalized mean intensity |
| `area` | Object area in pixels |
| `probability` | Detection confidence score (0–1) |
| `circularity` | Shape metric (1 = perfect circle) |

### Configuration

Edit `config/config.yaml` to adjust defaults for all CLI runs:

- **Hardware**: GPU device, batch size
- **Inspector**: SNR thresholds, denoising method
- **Aligner**: Feature detector type, transformation type, residual error thresholds
- **Analyst**: Detection model, area filters, colocalization settings
- **LLM**: API provider, model, temperature
