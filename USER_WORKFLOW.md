# SEA — Practical User Workflow Guide

This guide is for new users who want to run SEA from beginning to end in the GUI. Each step tells you **where to click**, **what you should see next**, and **where files land on disk** so you are not guessing.

You will follow this order:

1. Load data
2. (Optional) Image processing
3. Alignment
4. Exosome detection / segmentation
5. Results viewing
6. Data analysis report

---

## 1) Getting Started

### Start the app

1. **Open a terminal** and activate your conda environment, then start the API (default port **8765**):

```bash
conda activate SEA
python api_server_extended.py
```

- **What you should see:** the terminal stays attached to the process; leave this window open. If the port is busy, the process will error in the terminal—fix that before continuing.

2. **Open a second terminal**, go to the frontend folder, and start Vite:

```bash
cd frontend
npm run dev
```

- **What you should see:** a line like `Local: http://localhost:3000/`.

3. **In your web browser**, open **`http://localhost:3000`** (click the link in the terminal or paste the URL).

- **What you should see:** the SEA navigation bar at the top with tabs (e.g. **🤖 Agent Chat**, **🖼️ Image Processing**, …). All later steps assume this page is open.

### Organize your data first

Place TIFFs under the input folder by **sample** and **position** (folder names become the dropdown values in the app):

```
data/input/
└── <sample_id>/
    └── <position_id>/
        ├── *.tif
        └── *.tiff
```

- **Mouse:** after you add folders, use **🖼️ Image Processing** → **Sample** dropdown — new folders appear after a refresh (re-select tab or reload page if the list looks stale).
- **Naming:** channel discovery comes from the files in that position folder; exact TIFF filenames can vary as long as the backend recognizes them as channels for that position.

---

## 2) Tabs at a Glance

**How to switch tabs:** click the tab button in the top nav. SEA keeps all tabs mounted (hidden with CSS), so work in one tab is preserved when you switch away.

Use these **in order** for the core pipeline:

| Order | Tab | What you do there with the mouse |
|------:|-----|-----------------------------------|
| 1 | **🖼️ Image Processing** | Pick **Sample** → **Position** → **Load Position**; optional preprocessing per channel. |
| 2 | **🎯 Alignment** | Same load pattern; pick reference + method; **Detect Features** (non-manual) or draw **Manual Diagonal** boxes; **Run Alignment**; **Save**. |
| 3 | **🔬 Exosome Detection** | **Load Image**; pick **Channel** and **Method**; **Run Segmentation**; optional **Export Results** / RF training. |
| 4 | **📊 Results Viewer** | Load the same sample/position; browse files; **Run Colocalization Analysis**; download tables. |
| 5 | **📉 Data Analysis** | Choose CSV/Excel table(s) → **Run pipeline**; read the HTML report in-tab. |

**Optional (not required for the core pipeline):**

- **🤖 Agent Chat** — upload an example image and ask for parameter suggestions; you still apply numbers manually in **Image Processing**.

**Reuse between steps:** aligned TIFFs written under `data/input/<sample>/align/<position>/` are picked up automatically in **Exosome Detection** (green **Using aligned image** when applicable). Filter settings saved from Exosome can feed **Results Viewer** (see Step 3 and Step 4).

---

## 3) Step 1 — Load Data (Image Processing tab)

**Navigate:** click **🖼️ Image Processing** in the top bar.

### Load a position (required for everything else)

1. Under **📁 File Operations**, open the **Sample** dropdown and click a sample (e.g. `A2780Cis10`).
2. Open the **Position** dropdown and click a position (e.g. `P1`).
3. Click **Load Position** (enabled only when both are chosen).

**Immediately after click:** the button label switches to **Loading…** while the request runs. When it finishes:

- The main preview area fills with an image (when a default channel preview exists).
- The left sidebar shows channel-specific controls unlocked.
- **Reuse:** these loaded stacks are what **Alignment** and **Exosome Detection** read from disk for the same sample/position.

### Switch channels (mouse only)

- Use the **channel** control in the sidebar (dropdown or selector next to previews—labeled from backend channel metadata). **Click** another channel → the preview image updates to that channel’s current stage.

### Optional preprocessing (contrast, background, blur)

**Contrast enhancement**

1. Optionally tick **Apply to All Channels** (checkbox above the contrast button) if you want the same operation on every channel in one go.
2. Choose **CLAHE** or **Linear Stretch** from the **Method** dropdown under **Contrast Enhancement**.
3. Adjust sliders (clip limit, tile size, or percentiles) with the **mouse** by dragging; values update live in the labels.
4. Click **Apply Contrast Enhancement** (or **Apply to All Channels** when the checkbox is on).

**Immediately after click:** the button shows **Processing…** while the backend runs. When done, the **processed** preview updates; the app remembers per-channel **stage** (raw vs contrast-enhanced vs later steps).

**Background subtraction & Gaussian blur**

1. Click the **channel** selector so it matches the channel you want to edit (unless you rely on “apply to all” where offered).
2. Click **Apply Background Subtraction** or **Apply Gaussian Blur** as needed.

**While running:** same **Processing…** feedback on the active button.

**Reset**

- Click **Reset to Original** to revert the **selected** channel’s preprocessing state toward raw in the UI (see in-app behavior for scope). Use this if previews look wrong before moving to Alignment.

### “Load Image File” (optional sanity check)

- Click **Load Image File** → your browser’s file picker opens (TIFF/PNG/JPEG filtered).
- Pick a file → an **alert** shows only the **filename** (browsers do not expose full paths) and reminds you that SEA’s real workflow is **Sample / Position / Load Position**. This does **not** load that arbitrary file into the pipeline.

### What carries forward to later tabs

- Processed stages you create here appear under **Input stage** in **🎯 Alignment** (e.g. contrast-enhanced or later steps), so you can align on preprocessed data instead of raw.
- Folder layout remains `data/input/<sample>/<position>/` for raw inputs; preprocessing outputs are managed by the backend session for that position.

---

## 4) Step 2 — Alignment (Alignment tab)

**Navigate:** click **🎯 Alignment**.

### A) Load the same field of view

1. Under **📁 File Operations**, choose **Sample** and **Position** (same as Image Processing if you want consistency).
2. Click **Load Position**.

**Immediately after click:** **Loading…** on the button, then channel list, previews, and **Input stage** options populate. If a saved alignment snapshot exists for this sample/position, the app may **auto-restore** boxes/settings (see saved-state banner in the **Save / Restore** section).

### B) Choose what gets aligned

1. **Input stage** (dropdown in **📥 Input Stage**, visible after load): click **Raw** or a **preprocessed** stage name coming from Image Processing.
   - **Blue banner** reminds you that non-raw choices use Image Processing outputs.
2. **Reference Channel** (dropdown in **Method** area): **click** the channel that should stay fixed—other channels move to match it (⭐ marks the reference in lists).
3. **Channel selection** (checkboxes): **click** each channel you want in the solve (minimum **two**). Use **Select All** / **Deselect All** shortcuts if shown.
   - **Visual feedback:** status text shows `Selected: X / N`; a **red** warning appears if fewer than two channels are checked.

### C) Pick method and parameters (mouse)

- Open **Alignment Method** dropdown and click one of:
  - **Grid Intersection (Homography)** / **Grid Line Hough** / **Frequency Domain (FFT)** / **Manual Diagonal (Interactive)**.
- Open **Transform** dropdown: **Euclidean** vs **Affine** — click your choice.
- For grid/FFT methods, **drag sliders** and toggle **Invert** / FFT options as needed (controls are disabled while **Detect Features** or **Run Alignment** is running).

### D) Manual Diagonal — hands-on on the canvas

Use this when **Alignment Method** = **Manual Diagonal**.

**What the box means:** each channel gets a **rotated square** in image pixel space. Alignment uses the diagonal you draw to define that square’s placement and orientation for that channel relative to the **reference** channel’s box.

**Draw a new box (two clicks):**

1. In the channel list under **Manual Diagonal Boxes**, **click** a row (or use the preview **Channel** dropdown) so the editor is targeting the correct channel.
2. On the **preview canvas**, **left-click once** at the **first corner** of the diagonal (first anchor point, `P1`).
   - **Visual feedback:** a **dashed cyan line** “rubber-bands” from that point to the cursor until you finish.
3. **Left-click a second time** at the **opposite corner** of the diagonal (`P2`).
   - **Visual feedback:** the dashed line becomes a **solid yellow** rotated rectangle with **corner handles** and a **rotation handle** (small circle off the top edge).

**Edit an existing box (mouse + keyboard):**

- **Drag inside the box** → translate.
- **Drag a corner handle** → resize (opposite corner stays anchored during drag).
- **Drag the rotation handle** → change angle.
- **Arrow keys** → nudge the box (when the editor has focus; focus the canvas by clicking it).
- **Esc** → clear the current channel’s box (per editor behavior).

**Row tools:**

- **Copy Ref → All** — click once to duplicate the **reference** channel’s geometry to every selected channel (saves time when grids look identical).
- **Reset All** — clears all manual boxes.

**Ready rule:** a **green “ready to run”** style banner appears when **every selected channel** has a box. If the reference channel is missing a box, **Run Alignment** stays blocked for manual mode.

### E) Automated methods — Detect Features before Run

For **non-manual** methods:

1. Click **Detect Features**.
2. **While running:** the button disables / shows busy state; wait for completion.
3. **When done:** feature overlays and layer dropdowns populate; toggle **Show overlay in preview** and change **Preview Layer** to inspect lines/corners/FFT views.

### F) Run Alignment

1. Click **Run Alignment (N ch)** (label includes channel count).

**Immediately after click:** button disables during the solve. **On success:**

- Center **Stage** switches toward **aligned** previews; **Alignment QC** color overlay and **Shift Values** panel populate (Δx/Δy, identity warnings if a channel did not move).
- **On failure:** a browser **alert** shows the backend error string—read it, adjust channels/method/input stage, and try again.

**Outputs on disk (reuse in Exosome):** aligned TIFFs are written under:

`data/input/<sample>/align/<position>/<channel>_aligned.tif`

Exosome Detection prefers these per-channel mirrors when present (**Using aligned image**).

### G) Save / restore / export

- **Save** — persists the alignment snapshot (local + sample state) and, if **Download aligned TIFFs after save** is checked and a run exists, triggers export/download.
- **Load Saved** / **Reset** — recall or delete saved snapshot for this sample/position (see banners next to those buttons).

---

## 5) Step 3 — Exosome Detection / Segmentation (Exosome Detection tab)

**Navigate:** click **🔬 Exosome Detection**.

### A) Load the field and pick a channel

1. Left panel **Image Source:** choose **Sample** and **Position** from the dropdowns (same convention as other tabs).
2. Click **Load Image**.

**Immediately after click:** button shows a loading state; when finished, the center viewer shows the image, **Zoom: …%** appears, and **Channel** populates. **Double-click** the image area to **fit** the image to the viewport.

**Aligned vs raw (important reuse):** if alignment produced `…_aligned.tif` for the selected channel, green text **Using aligned image** appears near the channel selector—segmentation is running on the aligned stack, not raw.

### B) Global controls (all methods)

Under **Global Parameters**, drag sliders or type numbers for **Confidence Threshold**, **Min Area**, **Max Area**; toggle **Remove Small Objects** and **Fill Holes**. These feed the next **Run Segmentation** request.

### C) Method: SAM

1. **Method** dropdown → select **SAM (Segment Anything Model)**.
2. Fill **Checkpoint Path** (text field) with a valid model path on the machine running the backend; pick **Model Type** and **Device**.
3. **Detection Mode**:
   - **Box Prompt** — on the canvas, **press, drag, release** with the **left mouse button** to draw a rectangle (same gesture as dragging a box).
   - **Point Prompt** — **left-click** adds a **positive** point; **right-click** adds a **negative** point (browser context menu is suppressed on the viewer).
4. Click **Run Segmentation**.

**While running:** button reads **Running…**; wait until it returns.

**When done:** colored masks/outlines overlay the image; summary cards and **Detected Objects** table appear below. Use **Clear Prompts** (SAM only) to remove all box/point prompts without clearing detections.

**Export:** click **Export Results** when the table is non-empty — status text shows paths returned from the backend (artifacts for **Results Viewer**).

### D) Method: Blob

1. **Method** → **Blob-Based Detection**.
2. Adjust **Threshold**, **Min/Max Circularity**, **Min Inertia** sliders.
3. Click **Run Segmentation** and wait for **Running…** to finish.

**Result:** detections overlay + table; tune sliders and re-run to converge.

### E) Method: Random Forest — annotation loop (full detail)

**Brush painting (default “Exosome” mode in the radios):**

- **Left mouse button** drag → paint **positive (exosome)** labels (green brush visualization when “show marks” is on).
- **Right mouse button** drag → paint **background** labels (context menu is blocked on the viewer).
- If you switch the **Annotation Mode** radio to **Background (Right Click)** intentionally, **left** and **right** swap their default class roles—use the radio label as ground truth for which button paints which class.

**Brush size (three equivalent ways):**

- **Keyboard `[` and `]`** — decrease / increase brush radius by 1 (range **1–50 px**). Disabled while typing in a text field (`INPUT` / `TEXTAREA` / `SELECT` focused).
- **Shift + mouse wheel** over the viewer — same brush resize (wheel alone zooms—see below).
- **Brush Size** slider or numeric box in the left panel.

**Eraser (toggle with `E`):**

1. Press **`E`** (capital or lower-case) → **Eraser** mode toggles on (press **`E`** again to return to normal painting).
2. In eraser mode, **drag with left button** to remove **exosome** paint; **drag with right button** to remove **background** paint (matches the viewer tooltip: erase per-class strokes).

**Navigation on the canvas:**

- **Mouse wheel** → zoom in/out centered on cursor.
- **Shift + right-button drag** → **pan** the image (while held).
- **Double-click** viewer → **fit** image to viewport.

**Other RF controls:**

- **Clear Annotations** button — wipes all brush strokes **and** click history for this context (cannot undo once clicked).
- **Show brush strokes and prompt marks on image** — purely visual toggle; data stays in memory.

**Recommended RF workflow (annotate → segment → train → reuse):**

1. Paint rough **exosome** (left) and **background** (right) regions representative of the objects you care about.
2. Click **Run Segmentation**.
   - **While running:** **Running…** on the button.
   - **After:** inspect masks + **Detected Objects** table; toggle **Show Mask Outlines**, **Mask Opacity**, **Show Confidence Map** (RF) to judge quality.
3. If the run is acceptable as a first pass, optionally click **Train & save to disk** (enabled when annotations exist and not busy).
   - **After:** status text shows success/failure; **Saved path** displays the `.pkl` location. Models are saved per channel/position under the backend’s `rf_models/` layout (see Detailed UI Reference for exact naming); a **Saved model for inference** dropdown appears when multiple saves exist.
4. **Reuse on another position:** switch **Position** (and **Load Image** again). Pick a saved model from **Saved model for inference** and click **Run Segmentation** with **no** (or cross-FOV) strokes as appropriate—inference-only runs use the disk model path (see Detailed UI Reference §17 for the exact API split).

**Click History panel (right column, RF):** **Save** / **Load** / **Reset** syncs stored click lists per sample+position+channel+method; **Remove** on a row deletes that click. Use this when iterating across days.

**Filtered detections reuse:** in the results table, set numeric filters → **Apply** → **💾 Save Filter** so **Results Viewer** / downstream steps can respect that subset (see Step 4).

---

## 6) Step 4 — View Outputs and Colocalization (Results Viewer tab)

**Navigate:** click **📊 Results Viewer**.

### A) Load results for a sample/position

1. Choose **Sample** and **Position** (same dropdown pattern as other tabs).
2. Click **Load Position**.

**Immediately after click:** **Loading…** then the **ResultsVisualization** panel fills: file/artifact list, previews, download rows.

**If load fails:** an error panel appears with **Retry** — click it after fixing disk/network issues.

### B) Browse and download artifacts (mouse)

- **Click** a row or thumbnail in the outputs list (exact control labels mirror your build) to select an item.
- Click **Download** (per-item icon/button) or the larger **Download** under the preview when highlighted.
- **⬇️ Download Excel** (summary export) and **⬇️ Download Intensity Excel** (raw intensity extraction) are separate buttons—read the short description above each before clicking.

**Where files go:** browser default **Downloads** folder unless you configure the browser to ask each time.

### C) Reference-centered colocalization

Scroll to **Reference-Centered Colocalization**.

1. **Reference channel** dropdown — click the anchor channel (must be one of the exported channels).
2. **Marker channels** — tick checkboxes for each marker to compare against the reference (reference auto-removed from markers if re-selected).
3. **Analysis mode** — choose **Overlap** vs **Nearest Centroid**.
4. **Distance threshold (px)** — type a number; enabled only for **Nearest Centroid**.
5. Click **Run Colocalization Analysis**.

**Immediately after click:** label becomes **Running Analysis…**; wait.

**On success:** tables/plots refresh; a **green success** line may cite paths. **On error:** red **coloc-error** text explains the failure.

**Disk path (reuse):** colocalization artifacts are saved under:

`data/input/<sample>/results/<position>/`

Use these CSV/Excel exports as inputs to **📉 Data Analysis** (see Step 5 hint on the Data Analysis tab about column naming).

---

## 7) Step 5 — Data Analysis Report (Data Analysis tab)

**Navigate:** click **📉 Data Analysis**.

### A) Pick tables

1. Click the **Tables (CSV / Excel, multiple allowed):** file control (native file picker).
2. Select one `.csv`, `.xlsx`, or `.xls` file, or **Ctrl/Cmd-click** multiple compatible tables.

**Immediately after selection:** filenames appear as a bullet list under the control. If you change files after a run, the tab clears old errors/reports (returns to idle).

### B) Run the pipeline

1. Click **Run pipeline** (disabled until at least one file is chosen; also disabled while a run is active).

**Immediately after click:**

- If no files: red validation text **Please select at least one CSV or Excel file.**
- Otherwise: the header area shows a **spinner** panel: **Running analysis pipeline** (adds `(merging N files)` when `N>1`) and warns **not to close the tab**.

**On success:** spinner disappears; a large **white report card** (`cygnus-report-mount`) renders below with QC, marker summaries, plots (Plotly scripts execute in-page).

**On failure:** a red boxed **monospace** error shows backend validation details (column mismatches, merge issues, HTTP errors).

**Reuse:** export or copy tables from this HTML report for publications; source Excel/CSV can include columns such as `*_positive` and `*_score` as hinted in the tab subtitle.

---

## 8) Typical End-to-End Checklist

Use this as a **click-by-click** runbook; expand any step by jumping back to the section number.

1. **Image Processing (§3)**  
   Sample → Position → **Load Position** → (optional) contrast / blur / background buttons → verify previews per channel.

2. **Alignment (§4)**  
   Same sample/position → **Load Position** → pick **Input stage** + **Reference** + ≥2 channels → choose method → (**Detect Features** if not manual) **or** draw **Manual Diagonal** boxes (two left-clicks per channel) → **Run Alignment** → **Save** → confirm files under `data/input/<sample>/align/<position>/`.

3. **Exosome Detection (§5)**  
   **Load Image** → confirm **Using aligned image** when expected → pick **Method** → (RF: paint/erase, `[` `]` `E`, **Run Segmentation**, optional **Train & save to disk**) → **Export Results** → optional **💾 Save Filter** on the table.

4. **Results Viewer (§6)**  
   **Load Position** → inspect/download artifacts → configure colocalization controls → **Run Colocalization Analysis** → collect files from `data/input/<sample>/results/<position>/`.

5. **Data Analysis (§7)**  
   Attach exported tables → **Run pipeline** → read in-tab HTML report.

When a step **fails**, note the exact **alert** or red error text—§9 Troubleshooting maps common causes (unchanged reference section below).

---

## 9) Troubleshooting (Quick)

- **No samples/positions shown**
  - Check folder layout under `data/input/<sample>/<position>/`.

- **Run Alignment fails**
  - Reload position first, verify channels selected, try a different method or input stage.

- **No “Using aligned image” badge in Exosome Detection**
  - Re-run alignment and load position again in Exosome Detection.
  - Badge is channel-specific; it appears only if that channel has an aligned TIFF mirror.

- **Colocalization run succeeds but you cannot find files**
  - Look in `data/input/<sample>/results/<position>/`.

- **Data Analysis pipeline returns merge/column errors**
  - Ensure all uploaded files use the same required columns and marker set.

---

## 10) Optional: Agent Chat

If you want AI suggestions for preprocessing settings:

1. Open **🤖 Agent Chat**.
2. Upload an example image.
3. Ask for parameter recommendations.
4. Apply those settings in **Image Processing**.

This is optional and not required to complete the main workflow.

---

## Alignment Tab — Detailed UI Reference

This section is a control-by-control reference for the **🎯 Alignment** tab.

### Default state when the tab opens

- **Selected sample / position:** empty until loaded from available options.
- **Current stage (preview):** `aligned` (falls back to available stages once data is loaded).
- **Alignment method:** `Frequency Domain (FFT)`.
- **Transform:** `Euclidean`.
- **Feature overlay checkbox:** ON.
- **Crop mode (preview):** OFF.
- **TIFF auto-download after save:** OFF.
- **TIFF export format:** `Individual channels (zip)`.

---

### A) File Operations section

#### 1) `Sample` dropdown
- **What it controls:** which sample folder the tab works on.
- **Options:** values returned by backend input sample listing.
- **Disabled when:** no samples are available.
- **After change:** triggers position list refresh; may auto-select remembered/default position.

#### 2) `Position` dropdown
- **What it controls:** active position within selected sample.
- **Options:** values returned by backend for selected sample.
- **Disabled when:** no positions are available.

#### 3) `Load Position` button
- **Enabled when:** sample and position are selected and a load is not already running.
- **Disabled when:** missing sample/position or currently loading.
- **After click (success):**
  - Loads channel items and raw previews.
  - Loads available processed stages and picks latest available as input stage.
  - Loads processed previews/stats if available.
  - Initializes selected channels to all channels.
  - Sets preview stage to `processed` if processed previews exist, otherwise `raw`.
  - Resets feature/alignment runtime state.
  - Attempts to auto-restore saved alignment snapshot for this sample/position.

#### 4) `Load Image File` button
- **What it does:** opens Electron-native file picker only.
- **Disabled state:** not disabled in UI, but if Electron API is unavailable it shows an alert.
- **Important:** does **not** replace sample/position workflow; only displays selected path notice.

---

### B) Input Stage section

Shown only after a position is loaded.

#### 5) `Input stage` dropdown
- **What it controls:** which preprocessing output stage is sent to feature detection/alignment.
- **Options:** backend-provided stage keys (displayed as labels like Contrast Enhanced, etc.).
- **Disabled when:** alignment is running or feature detection is running.
- **When no options exist:** shows “No processed stages available” plus warning to process images first.

#### Status banner
- Blue banner indicates alignment input comes from Image Processing outputs.

---

### C) Channel Selection section

Shown only after loaded and channels exist.

#### 6) `Select All` button
- **Enabled when:** not aligning and not detecting.
- **After click:** marks every channel selected.

#### 7) `Deselect All` button
- **Enabled when:** not aligning and not detecting.
- **After click:** clears channel selection set.

#### 8) Per-channel checkbox rows
- **What each row controls:** include/exclude channel in alignment request payload.
- **Disabled when:** aligning or detecting.
- **After click/toggle:** updates selected channel set.
- **Visual states:**
  - reference channel row marked with ⭐ Ref styling.
  - selected rows highlighted.

#### Status box
- Shows “Selected: X / N channels”.
- Shows red warning when fewer than 2 channels are selected.

---

### D) Method section

Shown only when loaded.

#### 9) `Reference Channel` dropdown
- **Options:** all loaded channels.
- **Disabled when:** not loaded, aligning, or detecting.
- **Effect:** sets alignment anchor channel.

#### 10) `Alignment Method` dropdown
- **Options:**
  - Grid Intersection Homography
  - Grid Line Hough Transform
  - Frequency Domain (FFT)
  - Manual Diagonal (Interactive)
- **Disabled when:** aligning or detecting.
- **Effect:** switches available parameter controls and readiness rules.

#### 11) `Transform` dropdown
- **Options:**
  - Euclidean (rotation + translation)
  - Affine (+ shear + scale)
- **Disabled when:** aligning or detecting.

#### 12) Shared Grid Preprocessing controls  
Visible for **Grid Intersection** and **Grid Line Hough** methods.

- `Blur Kernel` slider (odd values 5–61, step 2)
  - Larger blur suppresses fine non-grid details.
- `Threshold Method` dropdown
  - Otsu (global), Adaptive Mean, Adaptive Gaussian.
- `Invert image` checkbox
  - Default ON; intended for dark grid lines.
- **All disabled when:** aligning or detecting.

#### 13) Grid Intersection (Homography) parameter sliders  
Visible only for this method.

- `Max Corners` (10–500)
- `Quality Level` (0.001–0.1)
- `Min Distance (px)` (2–60)
- `Block Size` (2–10)
- **Disabled when:** aligning or detecting.

#### 14) Grid Line Hough parameter sliders  
Visible only for this method.

- `Hough Threshold` (10–200)
- `Min Line Length (px)` (20–300)
- `Max Line Gap (px)` (1–80)
- `Angle Tolerance (°)` (2–30)
- `Merge Distance (px)` (5–80)
- **Disabled when:** aligning or detecting.

#### 15) Frequency Domain (FFT) controls  
Visible only for FFT method.

- `FFT Mode` dropdown:
  - Grid Structure Detection (full pipeline)
  - Simple Phase Correlation (fallback)
- `Window Function` dropdown:
  - Hann, Hamming, None
- Sliders:
  - `Min Grid Spacing (px)` (5–100)
  - `Max Grid Spacing (px)` (50–500)
  - `Peak Threshold` (0.05–0.9)
  - `Grid Line Width (px)` (1–10)
- `Enable Rotation Detection` checkbox
- **Disabled when:** aligning/detecting; some controls additionally disabled unless FFT mode is Grid Structure Detection.
- **Status warning:** Simple correlation mode shows aliasing-risk warning.

---

### E) Manual Diagonal Boxes section

Shown only when method is **Manual Diagonal** and data is loaded.

#### 16) Per-channel box status list
- **What it shows:** selected channels with ✅ if box exists, — if missing.
- **Row click:** switches preview channel to that row’s channel.
- **Reference channel:** marked with ⭐.

#### 17) `Copy Ref → All` button
- **Enabled when:** reference channel currently has a box.
- **After click:** clones reference box geometry into all channels.

#### 18) `Reset All` button
- **Enabled:** always in this section.
- **After click:** clears all manual boxes for all channels.

#### 19) Current box details panel
- Shows center, size, angle for current preview channel if a box exists.

#### 20) “Ready to run” banner
- Green banner appears only when every selected channel has a box.

---

### F) Feature Detection section

Shown when loaded, processed stages exist, and method is not Manual Diagonal.

#### 21) `Detect Features` button
- **Enabled when:** not detecting, not aligning, and an input stage is selected.
- **Disabled when:** currently detecting/alignment running/no input stage.
- **After click (success):**
  - Calls backend feature detection with selected method and params.
  - Stores `preview_layers`, feature counts, summary.
  - Sets stage to `features`.
  - Sets preview layer default to `detected`.
- **After click (failure):** shows red error banner with message.

#### 22) `Preview Layer` dropdown
- Appears after successful detection.
- **FFT options:** FFT Spectrum, Synthetic Grid, Intersections, All.
- **Grid options:** Preprocessed, Binary Mask, Detected (Lines/Corners), Intersections (Hough only).
- **Effect:** chooses which backend feature layer is rendered in center preview.

#### 23) `Show overlay in preview` checkbox
- Appears after successful detection.
- **Default:** ON.
- **Effect:** toggles whether feature layer or plain stage image is shown while in features stage.

#### Feature status indicators
- Green success block when detection is available.
- Per-channel feature count rows.
- Optional FFT aliasing warning in simple correlation mode.

---

### G) Run Alignment section

Shown when loaded.

#### 24) `Run Alignment (N ch)` button
- **Enabled when all are true:**
  - not aligning
  - not detecting
  - at least one processed stage exists
  - readiness rule passes:
    - manual diagonal: at least 2 selected channels and all selected channels have boxes
    - other methods: at least 2 selected channels and input stage selected
- **After click (success):**
  - Sends sample/position/input stage/ref/method/transform/selected channels (+ diagonal boxes for manual mode).
  - Writes aligned previews and stats to state.
  - Sets current stage to `aligned`.
  - Stores shift vectors and alignment run id.
  - Auto-switches preview channel to first non-reference channel that has aligned preview.
- **After click (failure):** alert with backend error.

#### Info banners in this section
- Manual mode hint when boxes are missing.
- Green “Features detected” note for non-manual methods.
- Blue tip when features not detected yet (optional step).
- Red warning when no processed stages exist.

---

### H) Crop section

Shown when loaded and aligned stage data exists.

#### 25) Collapsible `✂️ Crop` header
- Click to expand/collapse crop controls.
- “active” badge appears when crop rectangle exists.

#### 26) Numeric inputs: `X`, `Y`, `W`, `H`
- **Type:** number, min 0.
- **What they control:** crop rectangle in image pixel coordinates.
- **Effect:** updates crop rect used for export (non-destructive preview mask).

#### 27) `✅ Crop applied / No crop set` status button
- **Disabled when:** no crop rect.
- **Purpose:** status indicator only (not a submit action).

#### 28) `Clear` button
- **Disabled when:** no crop rect.
- **After click:** removes crop rectangle.

#### Crop warning
- Warning message shown if aligned stage is unavailable.

---

### I) Save / Restore section

Shown when loaded.

#### 29) Saved-state banner
- Green banner with timestamp if alignment snapshot exists in state.
- Yellow banner if none exists for this position.

#### 30) `Download aligned TIFFs after save` checkbox
- **Default:** OFF.
- **Disabled when:** alignment has not run (`alignmentRunId` is null).
- **Effect:** on Save, triggers backend TIFF export and browser download.

#### 31) TIFF format radio options (shown only when auto-download is ON)
- `Individual channels` (`zip`)
- `Multi-channel composite` (`composite`)
- **Effect:** sets export format used on Save.

#### 32) `Save` button
- **Enabled:** loaded state; no explicit disable.
- **After click:**
  - Saves snapshot to local storage and sample `.sea_state`.
  - Updates saved timestamp.
  - If auto-download enabled and alignment run exists, requests `/api/input/export_tiff` and downloads returned file.
  - Applies crop rect to export request if crop exists.

#### 33) `Load Saved` button
- **Enabled when:** snapshot key exists in local storage for selected sample/position.
- **After click:** restores saved diagonal boxes, shift vectors, reference channel, selected channels, method, crop rect, export preference, saved timestamp.

#### 34) `Reset` button
- **Enabled:** when loaded.
- **After click:**
  - Removes saved snapshot from local storage and `.sea_state` map for this position.
  - Clears boxes, shift vectors, alignment run id, crop rect, export status, saved timestamp.

#### 35) Export status line
- Appears when export/save operations set status.
- Color-coded:
  - blue info (in-progress),
  - green success (`✓`),
  - red failure/error.

---

### J) Center Preview controls

#### 36) `Crop` toggle button in preview header
- **Enabled when:** loaded and current stage is `aligned`.
- **Disabled when:** not loaded or not on aligned stage.
- **Effect:** toggles interactive drag-to-crop on preview image.

#### 37) `Stage` dropdown
- **Enabled when:** loaded.
- **Options shown dynamically based on available data:**
  - Raw
  - Processed
  - Feature Detection (if detected, non-manual)
  - Aligned (if available)
- **Effect:** changes displayed stage.

#### 38) `Channel` dropdown
- **Enabled when:** loaded and not in preview crop mode on aligned stage.
- **Options:** all loaded channels.
- **Effect:** changes displayed channel image.

#### 39) `Layer` dropdown (features stage only)
- Same layer choices as Feature Detection section.
- **Effect:** layer rendered in center preview.

#### 40) `Show overlay` checkbox (features stage only)
- **Effect:** show/hide feature overlay in center preview.

#### 41) Preview content area status messages
- “Load a sample and position to begin” when not loaded.
- “No preview available for stage/channel” when URL missing.
- In manual mode before aligned stage, shows ManualDiagonalEditor; after alignment and stage=aligned, shows aligned raster preview.

---

### K) Alignment QC: Color Overlay panel controls (inside Alignment tab)

#### 42) Overlay `Stage` dropdown
- **Enabled when:** at least one stage has channel data.
- **Options:** available stages only (`raw`, `contrast_enhance`, `step1`, `step2`, `step3`, `step4`, `aligned`).

#### 43) `Global Alpha` slider
- Range: 0.1–1.0, step 0.05.
- Controls blending intensity for channel overlays.

#### 44) `Background` buttons
- `Black` / `White`.
- Changes canvas background fill.

#### 45) Aligned-stage visualization toggles (visible only on `aligned`)
- `Show Shift Vectors` checkbox.
- `Show Before/After (Ghost)` checkbox.
- `Show Annotations` checkbox.
- If shift vectors enabled:
  - `Arrow Scale` slider (1.0–20.0, step 0.5)
  - `Base Length` slider (20–200 px, step 10)

#### 46) Channel controls in overlay
- `Select All` button: enables all overlay channels.
- `Deselect All` button: disables all overlay channels.
- Per-channel checkbox rows:
  - toggles channel visibility in overlay rendering.
  - row shows channel color and status labels (`[REF]`, `⚠ IDENTITY`, optional Δx/Δy).

#### 47) Overlay rendering indicators
- “Rendering overlay...” banner while compositing.
- Canvas dimension label shown under canvas.
- Help text explains interpretation of overlays and shift arrows.

---

### L) Right-side Alignment Shift Values panel

This panel is informational and interactive-by-row (not form inputs).

#### 48) No-data placeholder
- Shown until shift vectors are available.
- Message: run alignment first.

#### 49) Shift summary box
- Displays max shift, aligned channel count, and skipped/identity count.

#### 50) Per-channel cards
- Click card: triggers channel click callback (used for highlighting/visibility interactions).
- Hover card: triggers channel hover callback.
- Shows exact Δx, Δy, magnitude, optional rotation, residual error, and matches.
- Status pills:
  - `REF`
  - `ALIGNED`
  - `⚠ IDENTITY`

#### 51) Legend
- Static legend clarifying REF/ALIGNED/IDENTITY meanings.

---

## Exosome Detection Tab - Detailed UI Reference

This section documents all visible controls and indicators in the Exosome Detection tab (`frontend/src/components/ExosomeDetection.tsx`), including what each one does and when it is enabled.

### A) Left panel: Image Source

#### 1) `Sample` dropdown
- **Options:** `-- Select sample --` plus values returned from backend sample discovery.
- **Default:** empty.
- **Effect on change:**
  - saves current click history + annotations to the storage key for the **previous** sample/position/channel/method (so it is not lost),
  - updates selected sample,
  - clears aligned-image channel flags,
  - clears RF model status fields (`rfPersistedStatus`, `rfModelStatus`, `rfModelWarning`).

#### 2) `Position` dropdown
- **Options:** `-- Select position --` plus positions for selected sample.
- **Enabled when:** a sample is selected.
- **Default:** empty.
- **Effect on change:**
  - saves current click history + annotations for the **previous** position (same key scheme as above),
  - same status reset behavior as sample change, plus updates selected position.

#### 3) `Channel` dropdown (`exosome-channel-select`)
- **Options:** `-- Select channel --` plus loaded channel entries (display labels from backend item metadata).
- **Enabled when:** position has been loaded (`loaded=true`).
- **Default:** first channel from loaded position, or restored channel from saved UI state if valid.
- **Effect on change:**
  - saves current click history before switching channel,
  - clears prior segmentation artifacts (masks/scores/detections/probability map),
  - clears prompts (box/points),
  - clears RF status fields.

#### 4) `Using aligned image` indicator (green text near channel selector)
- **Shown when:** selected channel is present in `align_artifact_channels` returned by `/api/input/load_position`.
- **Meaning:** segmentation input resolves to saved aligned TIFF in sample align mirror path.
- **Hidden when:** no aligned artifact exists for current channel.

#### 5) `Load Image` button
- **Enabled when:** both sample and position are selected.
- **After click (`/api/input/load_position`):**
  - loads channel items and preview metadata,
  - sets loaded state and initial channel,
  - resets prompts/masks/detections/debug logs,
  - hides brush strokes and SAM prompt marks on the canvas (stored annotations and click history are **not** cleared; use the “Show brush strokes…” checkbox or draw again to see them),
  - loads crop-mode metadata when applicable,
  - attempts to load ground-truth points (`/api/exosome/ground_truth`),
  - fetches display image according to current display mode/LUT,
  - fits image to viewport,
  - in Random Forest mode, checks for persisted model and may auto-run inference.

#### 5b) `Show brush strokes and prompt marks on image` checkbox
- **Shown when:** a position has been loaded (`loaded=true`).
- **Effect:** toggles whether RF brush overlay, brush preview, and SAM box/point prompts are painted on the canvas (data in memory and storage is unchanged).

### B) Left panel: Image Info (read-only status)

#### 6) `Image Info` block
- **Shown when:** image is loaded.
- **Displays:** sample, position, crop-vs-normal mode, crop TIFF details, selected channel, normalization statistics, display-vs-detection source note, GT pixel size (if available).
- **Status banners inside block:**
  - crop mode badge (`Crop mode` vs `Normal`),
  - display pipeline panel with mode/LUT/range/stretch/cache details,
  - warning when non-default display mode is active,
  - confirmation when default `raw_16bit` mode is active.

### C) Left panel: Detection Method and method-specific controls

#### 7) `Method` dropdown
- **Options:** `SAM (Segment Anything Model)`, `Blob-Based Detection`, `Random Forest (Interactive, ML)`.
- **Default:** `Random Forest`.
- **Effect:** saves current click history + annotations for the previous method (per-channel storage key), then switches which parameter panels are rendered and which request payload is sent to `/api/exosome/segment`.

#### 8) Annotation Controls (Random Forest only)

##### `Annotation Mode` radio buttons
- **Options:** `Exosome (Left Click)`, `Background (Right Click)`.
- **Default:** `Exosome`.
- **Effect:** controls class label recorded while painting annotations on canvas.

##### `Brush Size` slider + number input
- **Range:** 1-50 px (integer).
- **Default:** 5.
- **Effect:** changes annotation brush radius for painting/erasing on canvas.

##### `Clear Annotations` button
- **Enabled:** always visible in RF mode.
- **After click:** clears both annotations and click history arrays.

#### 9) SAM Model Settings (SAM only)

##### `Checkpoint Path` text input
- **Accepts:** filesystem path string.
- **Default:** `checkpoints/pretrained/sam_vit_h_4b8939.pth`.
- **Validation behavior:** red border and required note shown when empty.
- **Runtime requirement:** segmentation blocks with alert if empty.

##### `Model Type` dropdown
- **Options:** `sam_vit_h`, `sam_vit_l`, `sam_vit_b` (labeled ViT-H/L/B).
- **Default:** `sam_vit_h`.

##### `Device` dropdown
- **Options:** `auto`, `cuda`, `cpu`.
- **Default:** `auto`.

#### 10) Blob Detection Parameters (Blob only)
- All are slider+number pairs:
  - `Threshold` (0-1, step 0.001, default 0.5),
  - `Min Circularity` (0-1, step 0.001, default 0.3),
  - `Max Circularity` (0-1, step 0.001, default 1.0),
  - `Min Inertia Ratio` (0-1, step 0.001, default 0.3).
- **Effect:** passed to blob segmentation request.

#### 11) Detection Mode (SAM only)
- **Options:** `Box Prompt`, `Point Prompt`.
- **Default:** `box`.
- **Runtime requirements:**
  - `Box Prompt` requires user-drawn box prompt.
  - `Point Prompt` requires at least one point prompt.
  - Missing prompt causes segmentation alert and request is not sent.

### D) Left panel: Global Parameters

#### 12) `Confidence Threshold` slider + number
- **Range:** 0-1, step 0.001.
- **Default:** 0.5.
- **Effect:** score/confidence threshold for SAM and Random Forest.

#### 13) `Min Area (px)` slider + number
- **Range:** 0-100000 (integer).
- **Default:** 10.
- **Effect:** minimum object area filter in backend segmentation.

#### 14) `Max Area (px)` slider + number
- **Range:** 0-1000000 (integer).
- **Default:** 10000.
- **Effect:** maximum object area filter in backend segmentation.

#### 15) `Remove Small Objects` checkbox
- **Default:** checked.
- **Effect:** toggles morphology cleanup flag in segmentation request.

#### 16) `Fill Holes` checkbox
- **Default:** unchecked.
- **Effect:** toggles fill/apply morphology behavior in segmentation request.

### E) Left panel: Actions

#### 17) `Run Segmentation` button
- **Enabled when:** image loaded and not currently running.
- **Disabled when:** not loaded or `isDetecting=true`.
- **Label changes to:** `Running...` during in-flight request.
- **After click:**
  - validates required context per method,
  - **Random Forest:** sends `POST /api/exosome/rf_model/load_and_segment` when a picker model path is set and either there are no annotations **or** the selected model was trained on a **different** position than the current FOV (cross-FOV inference; annotations are ignored in that case). Otherwise sends `POST /api/exosome/segment` (including in-memory train-from-annotations for preview when you have strokes on the current FOV).
  - **RF disk:** `Run Segmentation` never writes RF `.pkl` files; only **Train & save to disk** persists weights. Run trains in memory when using annotations on the current FOV (or loads a saved model for inference).
  - updates masks/scores/detections/probability map,
  - updates RF status/warnings when in RF mode,
  - refreshes RF persisted-model status.

#### 18) `Clear Prompts` button (SAM only)
- **Shown when:** detection method is SAM.
- **After click:** clears box prompt and point prompts only.

#### 19) `Export Results` button
- **Enabled when:** at least one detection exists.
- **Disabled when:** detection list is empty.
- **After click (`/api/exosome/export`):**
  - sends detections/masks/scores and settings,
  - sets export status (`Exporting...`, then success/failure),
  - populates exported file path list on success.

#### 20) `RF Model` subsection (Random Forest only)

Grouped below the main action buttons in a bordered panel.

##### `RF on disk` status card
- Shows whether saved model exists for selected sample/channel and last-saved timestamp/stem when available.
- Backend also returns `available_models` (per-position `ch*_<position>.pkl` and legacy layouts; one row per training position, newest `saved_at` wins; flat `ch*.pkl` is not listed — it is only the backward-compat default path) for the model picker.

##### `Saved model for inference` dropdown
- **Shown when:** more than one entry exists in `available_models`.
- **Default:** most recent save for the **current position** (from metadata), otherwise most recent overall.
- **Effect:** chooses which saved `.pkl` to use for `POST /api/exosome/rf_model/load_and_segment` when Run uses that path: **no annotations**, or **annotations present but the selected model was trained on another position** (inference only; strokes are ignored for that run). Same-position preview with brush strokes uses `/api/exosome/segment` (in-memory train, no disk save).

##### `Train & save to disk` button (primary / green)
- **Enabled when:** loaded, not detecting, annotation count > 0.
- **After click (`/api/exosome/rf_model/save`):**
  - trains from current annotations,
  - writes **per-position** `rf_models/<channel_stem>_<position>.pkl` (+ `.metadata.json`) and **copies** the same weights to the flat `ch*.pkl` default for backward compatibility,
  - updates saved-path/status text and refreshed persisted-model status.
- To refresh segmentation after saving, use **Run Segmentation** (or pick another saved model and Run for inference-only / cross-FOV runs).

##### RF feedback messages
- `rfModelStatus`: info/status text (train/saved/loaded/failure).
- `rfModelSavedPath`: shows model path when available.
- `rfModelWarning`: yellow warning box if backend returns model warning.

### F) Center panel: Viewer controls and overlays

#### 21) Display Mode radio group
- **Options:**
  - `ImageJ-like raw 16-bit` (`raw_16bit`, default),
  - `Enhanced stretch` (`enhanced`),
  - `Raw min→max` (`minmax`).
- **Important:** affects visualization only; detection always uses raw TIFF source.
- **Effect:** refreshes displayed preview through display endpoint/cache path.

#### 22) LUT radio group
- **Options:** `Grayscale` (default), `Red (ImageJ-like)`.
- **Effect:** changes viewer LUT for display image only.

#### 23) `Mask Opacity` slider + number
- **Range:** 0-1, step 0.01, default 0.5.
- **Effect:** adjusts overlay alpha for rendered masks in canvas.

#### 24) `Show Mask Outlines` checkbox
- **Default:** checked.
- **Effect:** toggles mask boundary rendering.

#### 25) `Show Filtered Only` checkbox
- **Enabled when:** detections exist.
- **Disabled when:** no detections.
- **Default:** unchecked.
- **Effect:** hides non-filter-matching detections from overlay/table-linked display.
- **Label count:** shows `(filtered / total)` when detections exist.

#### 26) `Guide from Ref` checkbox
- **Enabled when:** current channel is not `C0` and guide reference points are available (from saved filtered C0 detections).
- **Disabled when:** on `C0`, or guide data unavailable.
- **Default:** unchecked.
- **Effect:** overlays reference-guide points for cross-channel comparison.

#### 27) `Show Confidence Map` checkbox (RF only)
- **Default:** unchecked.
- **Effect:** toggles visualization of RF probability map when available.

#### 28) `Show Ground Truth` checkbox
- **Enabled when:** ground-truth points are loaded.
- **Disabled when:** no GT points found.
- **Default:** unchecked.
- **Label annotations:**
  - shows `(N pts)` when GT file loaded,
  - shows `(no CSV found)` when image loaded but GT file missing.

#### 29) `GT Debug Mode` checkbox (visible when GT is shown)
- **Shown when:** `Show Ground Truth` is on and GT points exist.
- **Default:** unchecked.
- **Effect:** enables debug rendering/logging aids for GT validation (magenta markers/labels + nearest-point console output).

#### 30) Viewer status indicators
- `Zoom: XX%` badge appears on loaded image.
- bottom source label clarifies `Display` image vs `Detection` source.
- placeholder text shown when no image is loaded: `Load an image to start detection`.

### G) Results region under viewer (visible after detections)

#### 31) Summary cards
- **Shown when:** detections exist.
- **Displays:** object count, mean/median area, min/max area.

#### 32) Area Distribution chart
- **Shown when:** detections exist.
- **Behavior:** auto-selects log scale for highly skewed area distributions.

#### 33) Detected Objects table controls

##### Filter inputs (text inputs)
- `Area min`, `Area max`, `Circ min`, `Circ max`.
- **Accepts:** numeric text; blank means no bound.
- **Effect:** values are applied when `Apply` button is clicked.

##### `Apply` button
- Applies current area/circularity filters and refreshes filtered index set.

##### `💾 Save Filter` button
- Persists filtered object IDs keyed by sample+position+channel for use in Results Viewer workflows.
- Shows blue status message on success/failure text path.

##### `Reset` button
- Clears all four filter fields.

##### Sortable columns
- Click table headers for `ID`, `Area`, `Perimeter`, `Circularity` to toggle sort key/direction.
- Clicking a row selects object and highlights it in canvas overlay.

#### 34) Export Status block
- **Shown when:** export action has produced status text.
- **Displays:** status string and exported file paths list (if provided by backend).

### H) Right panel: Annotation Tools

#### 35) Debug & Calibration toggles

##### `Debug coordinate logging` checkbox
- **Default:** unchecked.
- **Effect:** records click/transform debug entries to `debugLogs`.

##### `Calibration Mode (show crosshairs)` checkbox
- **Default:** unchecked.
- **Effect:** enables calibration visualization mode in viewer.

##### Debug log action buttons (shown when debug logging enabled)
- `Clear Logs`: empties debug log array.
- `Copy JSON`: copies current debug logs JSON to clipboard.

##### Debug log list panel
- Shows live log entries with timestamp and coordinate transform details.
- Shows placeholder `No debug logs yet...` when empty.

#### 36) Pixel Inspector (RF only)
- **Shown only in Random Forest mode.**
- Displays current clicked pixel coordinates/intensity.
- After segmentation with RF probability output, also shows confidence and predicted class.
- Shows instruction text when no pixel has been clicked.

#### 37) Click History toolbar (RF only)
- `Save`: saves click history + annotations to storage key.
- `Load`: restores click history + annotations from storage.
- `Reset`: clears history and removes stored entry.
- `JSON`: copies click history JSON to clipboard.
- `CSV`: copies click history as CSV to clipboard.
- **Context sync:** when sample, position, channel, or detection method changes, the current context is saved to storage (same key as `sea_clicks_<sample>_<position>_<channel>_<method>`), then the panel reloads from storage for the new context (empty lists if nothing is saved for that key). Previous contexts remain on disk.

#### 38) Click History grouped lists (RF only)
- Two panels: `Exosome` and `Background`, each with count.
- Each entry shows coordinates, timestamp, intensity, confidence (if present), class.
- `Remove` button per entry removes that click and its linked annotation (if any).

---

## Data Analysis Tab - Detailed UI Reference

This section documents all visible UI elements in the Data Analysis tab (`frontend/src/components/DataAnalysis.tsx`) and exactly what each one does.

### A) Header and file selection controls

#### 1) Tab title (`Data Analysis`)
- Static heading only.
- No interaction.

#### 2) `Tables (CSV / Excel, multiple allowed)` file input
- **Type:** file picker (`type="file"`).
- **Accepts:** `.csv`, `.xlsx`, `.xls` (and matching MIME types).
- **Multiple selection:** enabled (`multiple` attribute).
- **Enabled when:** pipeline is not running.
- **Disabled when:** `Run pipeline` is in progress (`phase === 'running'`).
- **After file selection change:**
  - updates selected file list from the browser file input,
  - clears validation and error messages,
  - if the tab previously showed success/error output, resets state to idle and clears previous report display.

#### 3) Selected-files list / empty-state text
- **If one or more files selected:** shows a bullet list of filenames.
- **If none selected:** shows `No files selected.`
- Informational only; not interactive.

#### 4) `Run pipeline` button
- **Enabled when:** at least one file is selected and not currently running.
- **Disabled when:**
  - no file selected, or
  - a run is currently in progress (`phase === 'running'`).
- **After click (`runPipeline`):**
  1. validates that at least one file is selected; if not, shows validation hint and aborts.
  2. clears existing hints/errors/report content.
  3. sets phase to running.
  4. appends all selected files to `FormData` using key `files`.
  5. sends `POST` request to `/api/cygnus/run`.
  6. handles responses:
     - non-JSON response: sets error with HTTP code,
     - non-OK or `success=false`: shows backend message (or fallback HTTP failure text),
     - missing `report_html`: shows explicit error,
     - success with valid `report_html`: stores HTML and switches to done state.
  7. network/client exception: shows `Network or client error: ...` and switches to error state.

### B) Inputs, toggles, selectors present in this tab

#### 5) Dropdowns/selectors
- None in this tab.

#### 6) Checkboxes/toggles/radio buttons
- None in this tab.

#### 7) Text/number input fields
- No free-text or numeric form fields are present.
- The only user input control is the file picker.

### C) Status indicators and feedback panels

#### 8) Validation hint (red text)
- **Shown when:** user tries to run without selecting files.
- **Message:** `Please select at least one CSV or Excel file.`
- **Cleared when:** file selection changes or a new run starts.

#### 9) Running panel (spinner + status message)
- **Shown when:** `phase === 'running'`.
- **Content:** spinner and message:
  - always starts with `Running analysis pipeline`.
  - if more than one file selected, appends `(merging N files)`.
  - includes warning not to close the tab during execution.

#### 10) Error panel (red boxed area)
- **Shown when:** `phase === 'error'` and `errorMessage` is non-empty.
- **Displays:** backend validation messages, HTTP failures, non-JSON response errors, or network/client exceptions.
- **Formatting:** monospaced, preserves line breaks (`white-space: pre-wrap`) for readable backend error details.

#### 11) Report output mount (`cygnus-report-mount`)
- **Shown when:** `phase === 'done'` and `reportHtml` exists.
- **Behavior after successful run:**
  - parses returned HTML via `DOMParser`,
  - injects report `<style>` tags into scoped container,
  - injects report body HTML,
  - re-creates `<script>` nodes so embedded scripts execute (needed for interactive report assets like Plotly),
  - clears/rebuilds content when report changes or component unmounts.

### D) State behavior that affects user-visible UI

#### 12) Phase transitions and visible states
- `idle`: file picker + run button + file list/empty state.
- `running`: shows loading panel; blocks changing files and re-running.
- `error`: shows error panel.
- `done`: shows rendered report panel.

#### 13) What resets previous output
- Selecting/changing files after a completed or failed run resets the tab from `done/error` back to `idle` and removes old report output, so a new run starts from a clean state.
