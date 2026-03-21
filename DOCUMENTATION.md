# 📚 SEA Complete Documentation

**Last Updated:** January 2026  
**Version:** 1.0

This document consolidates all documentation for the SEA (Semiautomated Exosome Analysis) pipeline, including installation, usage, troubleshooting, and all recent fixes.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Pipeline Overview](#pipeline-overview)
4. [Visualization GUI](#visualization-gui)
5. [Agent Chat](#agent-chat)
6. [Preprocessing Workflow](#preprocessing-workflow)
7. [Troubleshooting & Fixes](#troubleshooting--fixes)
8. [Configuration](#configuration)
9. [API Reference](#api-reference)

---

## 🚀 Quick Start

### 30-Second Start
```bash
# 1. Activate environment
conda activate SEA

# 2. Run pipeline
python pipeline.py --samples 1

# 3. View results
ls data/output/1/
```

### Start GUI (3 Terminals)
```bash
# Terminal 1: Main Backend
python api_server_extended.py  # Port 5000

# Terminal 2: Agent Chat
python api_agent_chat.py      # Port 5001

# Terminal 3: Frontend
cd frontend && npm run dev     # Port 3000

# Open: http://localhost:3000
```

---

## 📦 Installation

### 1. Create Conda Environment
```bash
conda env create -f environment.yaml
conda activate SEA
```

### 2. Install Optional Dependencies
```bash
# Noise2Void (if needed)
pip install n2v --no-deps

# Image codecs (for LZW TIFF)
pip install imagecodecs
```

### 3. Configure OpenAI API (Optional)
```bash
export OPENAI_API_KEY="your-api-key-here"
# Or add to ~/.bashrc
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import kornia, stardist; print('Dependencies OK')"
```

---

## 🔬 Pipeline Overview

### Architecture
SEA uses a **Hybrid Architecture**:
- **Local SOTA Models** for pixel-level tasks (denoising, registration, segmentation)
- **LLMs** (via API) for high-level reasoning and quality assessment

### Three Phases

#### Phase 1: Inspector Agent
- Image quality assessment (SNR calculation)
- Denoising (Noise2Void/CARE)
- Anchor channel detection (sharpest channel)

#### Phase 2: Aligner Agent
- Multi-channel registration using SuperPoint + SuperGlue
- GPU-accelerated warping with Kornia
- Affine transformation (with TPS fallback)
- LLM visual QA (optional)

#### Phase 3: Analyst Agent
- Exosome detection with StarDist
- Colocalization analysis
- CSV export with coordinates and intensities
- LLM statistical interpretation (optional)

### Input Data Structure
```
data/input/
├── Sample_01/
│   ├── ch1_CD63.tif
│   ├── ch2_CD81.tif
│   └── ch3_Syntenin.tif
└── Sample_02/
    └── ...
```

### Output Structure
```
data/output/
└── Sample_01/
    ├── preprocessed/          # Denoised images (.png)
    ├── registered/            # Aligned images (.png)
    ├── labels/                # Detection labels (.png)
    ├── Sample_01_overlay.png # RGB overlay visualization
    ├── Sample_01_detections.csv # Quantification data
    └── Sample_01_plots.png    # Statistical plots (if LLM enabled)
```

### CSV Output Format
- `channel`: Channel name
- `object_id`: Unique object identifier
- `x_coord`, `y_coord`: Pixel coordinates
- `intensity`: Normalized intensity value
- `area`: Object area in pixels
- `probability`: Detection confidence
- `colocalized`: Boolean (True if colocalized with another channel)

---

## 🖥️ Visualization GUI

### Features
- **3-Step Workflow**: Before/After comparison → Movement analysis → Final results
- **Interactive**: Frame selection, sortable tables, image preview
- **Colocalization Display**: Per-channel statistics and rates
- **Pipeline Control**: Start/stop pipeline, real-time logs, progress tracking

### Quick Start
```bash
# Setup (one time)
conda activate SEA
bash setup_visualization.sh

# Run (3 terminals)
# Terminal 1: python api_server_extended.py
# Terminal 2: python api_agent_chat.py
# Terminal 3: cd frontend && npm run dev
```

### Tabs

#### 1. Agent Chat (Tab 1)
- Upload images for AI analysis
- Get preprocessing recommendations
- Approve/edit parameters
- Auto-save to manifest

#### 2. Pipeline Control (Tab 2)
- Start/stop pipeline execution
- Select samples and steps
- Real-time progress and logs
- WebSocket streaming

#### 3. Results Viewer (Tab 3)
- View alignment results
- Colocalization statistics
- Download output files
- Before/after comparisons

---

## 🤖 Agent Chat

### Purpose
AI-powered preprocessing assistant that:
- Analyzes uploaded images
- Recommends numeric preprocessing parameters
- Provides reasoning for recommendations
- Outputs structured JSON for pipeline integration

### Features
- **Image Upload**: Supports TIFF (auto-converts to PNG for OpenAI)
- **GPT-4o Vision**: Analyzes images when provided
- **Auto-Save**: Parameters automatically saved to manifest
- **Human-in-the-Loop**: Approve, edit, or re-run recommendations
- **TXT Export**: Creates readable parameter files alongside images

### Workflow
1. Upload image in Agent Chat tab
2. Ask: "What preprocessing does this image need?"
3. Agent analyzes and recommends parameters
4. Parameters auto-saved to `outputs/runs/<run_id>/preprocess_manifest.json`
5. Click "Approve" or "Edit" to finalize
6. TXT file created in same directory as image

### API Endpoints
- `POST /api/agent/chat` - Chat with agent
- `POST /api/agent/manifest/<run_id>/image/<id>/approve` - Approve parameters
- `POST /api/agent/manifest/<run_id>/image/<id>/edit` - Edit parameters
- `POST /api/agent/preprocess/<run_id>` - Run preprocessing

---

## 🔄 Preprocessing Workflow

### Complete Flow
1. **Agent Chat** → Upload image, get recommendations
2. **Auto-Save** → Parameters saved to manifest
3. **Human-in-the-Loop** → Approve/edit parameters
4. **Preprocessing** → Apply parameters to images
5. **Results** → View before/after comparisons

### Manifest Structure
```json
{
  "run_id": "agent_session",
  "images": {
    "image_id": {
      "status": "approved",
      "recommended_params": {...},
      "final_params": {...},
      "reasoning": "...",
      "confidence": 0.85
    }
  }
}
```

### Supported Parameters
- `denoising_strength` (0.0-1.0)
- `background_subtraction_radius` (1-500)
- `clahe_clip_limit` (0.1-10.0)
- `clahe_grid_size` (2-32)
- `threshold_value` (0-255)
- `gamma_correction` (0.1-5.0)
- `sharpening_strength` (0.0-2.0)

---

## 🔧 Troubleshooting & Fixes

### Issue: Zero Colocalization Despite Visual Evidence

**Root Causes:**
1. Coordinate format was wrong (y, x instead of x, y)
2. Matching algorithm too restrictive (one-to-one only)
3. CSV marking code had `pass` (didn't save results)

**Fixes Applied:**
- ✅ Fixed coordinate format for `cdist` calculation
- ✅ Improved matching algorithm (many-to-many with greedy matching)
- ✅ Fixed CSV marking code to actually save colocalization
- ✅ Added channel names to pairs for proper marking

**Expected Results:**
- Before: 0 colocalized pairs
- After: 100-180 colocalized pairs (for sample 1)

### Issue: ch5 Only Detected 26 Cells (Should Be 50-100+)

**Root Causes:**
1. Poor image normalization (simple scaling lost contrast)
2. Probability threshold too high (default 0.5)
3. No percentile normalization (StarDist best practice)

**Fixes Applied:**
- ✅ Added percentile normalization (1st-99th percentile)
- ✅ Lowered probability threshold to 0.3 (configurable)
- ✅ Direct uint16 handling (no information loss)

**Expected Results:**
- Before: 26 detections
- After: 50-100+ detections (2-4x increase)

### Issue: Segmentation Not Appropriate

**Potential Causes:**
1. Wrong StarDist model (`2D_versatile_fluo` for DAPI nuclei)
2. Image normalization not optimal
3. Probability threshold too strict

**Recommended Solutions:**
- Switch to `2D_paper_dsb2018` model (better for DAPI/brightfield)
- Use percentile normalization (already fixed)
- Lower probability threshold (already fixed)

### Issue: WebSocket Connection Spam / Unstable Connection

**Symptoms:**
- Status indicator flickering between "Connected" and "Disconnected"
- Hundreds of WebSocket connection attempts in server logs
- UI becomes unresponsive

**Root Causes:**
1. `useEffect` in `useWebSocket` had `connect`/`disconnect` in dependency array (causing infinite loop)
2. Callbacks recreated on every render, triggering reconnections
3. No protection against multiple simultaneous connections

**Fixes Applied:**
- ✅ Removed `connect`/`disconnect` from `useEffect` dependencies (only depend on `url`)
- ✅ Used refs for callbacks to prevent function recreation
- ✅ Added connection state tracking to prevent multiple connections
- ✅ Only reconnect on non-manual closes (code !== 1000)
- ✅ Memoized callbacks in `PipelineControl.tsx`
- ✅ Improved backend error handling for disconnections

**Result:** Stable WebSocket connection, no flickering, single connection per tab

### Issue: Images Saved as TIFF Instead of PNG

**Fix:**
- Modified `pipeline.py` to save as `.png`
- Updated `utils/image_utils.py` for PNG support
- `api_server_extended.py` serves PNG directly

---

## ⚙️ Configuration

### Key Settings (`config/config.yaml`)

#### Hardware
```yaml
hardware:
  device: "cuda"
  gpu_id: 0
  batch_size: 4
```

#### Inspector
```yaml
inspector:
  snr_threshold: 10.0
  denoise_threshold: 5.0
  denoise_method: "n2v"
```

#### Aligner
```yaml
aligner:
  feature_detector: "superpoint"
  matcher: "superglue"
  initial_transform: "affine"
  fallback_transform: "tps"
```

#### Analyst
```yaml
analyst:
  detection_model: "stardist"
  stardist_model_type: "2D_versatile_fluo"  # Or "2D_paper_dsb2018" for DAPI
  stardist_prob_thresh: 0.3  # Lower = more detections
  min_object_size: 1
  max_object_size: 5000
  intensity_threshold: 0.05
  colocalization_distance: 10.0  # Pixels
```

---

## 🌐 API Reference

### Main Backend (`api_server_extended.py` - Port 5000)

#### Results Viewer
- `GET /api/alignment/<sample_name>` - Get alignment results
- `GET /api/image/<sample_name>/<image_type>/<filename>` - Get image
- `GET /api/result/<sample_name>/<filename>` - Get result file

#### Pipeline Control
- `POST /api/pipeline/start` - Start pipeline
- `POST /api/pipeline/stop` - Stop pipeline
- `GET /api/pipeline/status` - Get status
- `GET /api/pipeline/logs` - Get logs
- `WS /ws/pipeline` - WebSocket for real-time updates

### Agent Chat (`api_agent_chat.py` - Port 5001)

- `POST /api/agent/chat` - Chat with agent
- `POST /api/agent/manifest/<run_id>/image/<id>/approve` - Approve parameters
- `POST /api/agent/manifest/<run_id>/image/<id>/edit` - Edit parameters
- `POST /api/agent/preprocess/<run_id>` - Run preprocessing
- `GET /api/agent/runs` - List runs
- `GET /api/agent/health` - Check API key

---

## 📊 Common Adjustments

### Too Few Detections?
```yaml
analyst:
  intensity_threshold: 0.01  # Lower from 0.05
  stardist_prob_thresh: 0.2   # Lower from 0.3
```

### Too Many False Positives?
```yaml
analyst:
  intensity_threshold: 0.1   # Raise from 0.05
  morphology_filter: true     # Enable shape filtering
```

### Memory Issues?
Edit `agents/aligner.py`:
```python
max_size = 512  # Reduce from 1024
```

---

## 🧪 Testing

### Test Pipeline
```bash
python pipeline.py --samples 1
```

### Test Colocalization
```bash
python test_colocalization_diagnostic.py 1
```

### Test Detection Counts
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/output/1/1_detections.csv')
print(f'Total: {len(df)}')
print(f'Colocalized: {(df[\"colocalized\"] == True).sum()}')
print(df['channel'].value_counts())
"
```

---

## 📝 File Structure

```
SEA/
├── agents/
│   ├── inspector.py      # Phase 1: Quality assessment
│   ├── aligner.py       # Phase 2: Registration
│   └── analyst.py       # Phase 3: Detection & analysis
├── core/
│   ├── manifest/        # Preprocessing parameter storage
│   └── preprocess/      # Preprocessing engine
├── config/
│   └── config.yaml      # Configuration file
├── frontend/            # React TypeScript GUI
├── api_server_extended.py  # Main backend (port 5000)
├── api_agent_chat.py      # Agent Chat backend (port 5001)
└── pipeline.py          # Main pipeline controller
```

---

## 🎯 Quick Reference

### Which Servers to Run?
```bash
# Terminal 1: Main Backend (ALL features)
python api_server_extended.py  # Port 5000

# Terminal 2: Agent Chat
python api_agent_chat.py      # Port 5001

# Terminal 3: Frontend
cd frontend && npm run dev     # Port 3000
```

**Don't run:** `api_server.py` (old version, use `api_server_extended.py` instead)

### Status Codes
- `0` = ✅ Success
- `1` = ⚠️ Some samples failed
- `2` = ❌ Fatal error

---

## 📚 Additional Resources

- **Main README**: `README.md` - Project overview
- **Frontend README**: `frontend/README.md` - Frontend details
- **Core Modules**: See `core/` directory for implementation details

---

## ✅ Recent Fixes Summary

1. **Colocalization Fixes** (January 2026)
   - Fixed coordinate format
   - Improved matching algorithm
   - Fixed CSV marking code

2. **ch5 Detection Fix** (January 2026)
   - Added percentile normalization
   - Lowered probability threshold
   - Direct uint16 handling

3. **PNG Output Format** (January 2026)
   - Changed from TIFF to PNG for web compatibility

4. **Agent Chat** (January 2026)
   - Complete OpenAI integration
   - Image upload with TIFF support
   - Auto-save parameters
   - TXT file export

5. **Preprocessing Workflow** (January 2026)
   - Manifest-based storage
   - Human-in-the-loop approval
   - Batch preprocessing

---

**For detailed information on specific topics, see the code comments and inline documentation.**


