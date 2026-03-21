# SEA: Exosome Analysis Pipeline

**SEA** (Semiautomated Exosome Analysis) is an AI Agent Pipeline for multi-channel fluorescence exosome imaging analysis. The pipeline automates image registration, denoising, detection, and quantification of exosomes across multiple fluorescence channels.

## Architecture

SEA uses a **Hybrid Architecture** approach:
- **Local SOTA Deep Learning Models** for pixel-level precision tasks (denoising, registration, segmentation)
- **LLMs** (via API) for high-level reasoning, quality assessment, and statistical interpretation

### Pipeline Phases

1. **Inspector Agent** - Image quality assessment, denoising (Noise2Void/CARE), anchor channel detection
2. **Aligner Agent** - Multi-channel registration using SuperPoint + SuperGlue, GPU-accelerated warping with Kornia
3. **Analyst Agent** - Exosome detection with StarDist, colocalization analysis, CSV export, statistical interpretation

## Requirements

- **Hardware**: NVIDIA GPU (RTX 5090 recommended) with CUDA support
- **OS**: Linux (tested on Ubuntu)
- **Python**: 3.10
- **Conda**: For environment management

## Installation

### 1. Create Conda Environment

```bash
# Navigate to project directory
cd /home/mhl/Project_SM/SEA

# Create environment from YAML
conda env create -f environment.yaml

# Activate environment
conda activate SEA
```

### 2. Post-Installation: Install N2V (Optional)

Due to a dependency conflict between StarDist and N2V (both require different csbdeep versions), N2V must be installed separately:

```bash
# After activating the environment
conda activate SEA

# Install N2V without dependencies (uses existing csbdeep)
pip install n2v --no-deps

# OR install with compatible csbdeep version (if needed)
# pip install csbdeep==0.7.2 n2v
```

**Note:** If you don't need N2V denoising, you can skip this step. The pipeline will use rule-based denoising decisions instead.

### 3. Verify Installation

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check key dependencies
python -c "import kornia; import stardist; print('Dependencies OK')"

# Test N2V (if installed)
python -c "import n2v; print('N2V OK')" 2>/dev/null || echo "N2V not installed (optional)"
```

### 4. Install imagecodecs (for LZW-compressed TIFF files)

If your TIFF files use LZW compression, install imagecodecs:

```bash
conda activate SEA
pip install imagecodecs
```

**Note:** This is already included in the environment.yaml, but if you created the environment before this update, install it manually.

### 5. RTX 5090 Users: CUDA Compatibility Warning

If you're using an RTX 5090, you may see a CUDA compatibility warning. **This is normal and the pipeline will still work.** See `RTX5090_NOTES.md` for details.

### 6. Configure LLM API (Optional)

For LLM-based quality assessment and statistical analysis, set your API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or add to your `~/.bashrc`:
```bash
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
```

## Input Data Structure

Organize your data as follows:

```
INPUT/
├── Sample_01/
│   ├── ch1_CD63.tif
│   ├── ch2_CD81.tif
│   └── ch3_Syntenin.tif
├── Sample_02/
│   ├── ch1_CD63.tif
│   └── ch2_CD81.tif
└── ...
```

**Important:**
- Images must be 16-bit or 32-bit grayscale TIFF files
- Each sample should be in its own subdirectory
- Channel files can have any naming convention (detected by `.tif`/`.tiff` extension)

## Usage

### Basic Usage

```bash
# Process all samples in INPUT directory
python pipeline.py

# Process specific samples
python pipeline.py --samples Sample_01 Sample_02

# Use custom input directory
python pipeline.py --input /path/to/your/data

# Use custom config file
python pipeline.py --config config/custom_config.yaml
```

### Programmatic Usage

```python
from pipeline import ExosomeAnalysisPipeline
from pathlib import Path

# Initialize pipeline
pipeline = ExosomeAnalysisPipeline()

# Run on all samples
results = pipeline.run()

# Or process specific sample
sample_dir = Path("data/input/Sample_01")
result = pipeline.process_sample(sample_dir)
```

## Output Structure

For each processed sample, the pipeline generates:

```
data/output/
└── Sample_01/
    ├── preprocessed/          # Denoised images
    │   ├── ch1_CD63_preprocessed.tif
    │   └── ch2_CD81_preprocessed.tif
    ├── registered/             # Aligned images
    │   ├── ch1_CD63_registered.tif
    │   └── ch2_CD81_registered.tif
    ├── labels/                 # Detection labels
    │   ├── ch1_CD63_labels.tif
    │   └── ch2_CD81_labels.tif
    ├── Sample_01_overlay.png   # RGB overlay visualization
    ├── Sample_01_detections.csv # Quantification data
    └── Sample_01_plots.png     # Statistical plots (if LLM enabled)
```

### CSV Output Format

The `*_detections.csv` file contains:
- `channel`: Channel name
- `object_id`: Unique object identifier
- `x_coord`, `y_coord`: Pixel coordinates
- `intensity`: Normalized intensity value
- `area`: Object area in pixels
- `probability`: Detection confidence
- `circularity`: Morphological circularity metric

## Configuration

Edit `config/config.yaml` to customize:

- **Hardware settings**: GPU/CPU, batch size
- **Inspector**: SNR thresholds, denoising method
- **Aligner**: Feature detector, transformation type, residual thresholds
- **Analyst**: Detection model, filtering parameters
- **LLM**: API provider, model selection, temperature

## Key Features

### Phase 1: Inspector
- Automatic SNR calculation
- LLM-based quality assessment
- Noise2Void/CARE denoising
- Anchor channel detection (sharpest channel)

### Phase 2: Aligner
- SuperPoint feature detection (robust for dot-like structures)
- SuperGlue matching
- Affine transformation (preserves data integrity)
- Thin-Plate Spline fallback (for lens distortion)
- GPU-accelerated warping with Kornia
- LLM visual QA (optional)

### Phase 3: Analyst
- StarDist object detection (excellent for overlapping round objects)
- Morphology and intensity filtering
- Colocalization analysis
- CSV export with coordinates and intensities
- LLM statistical interpretation (optional)

## Troubleshooting

### CUDA Issues
```bash
# Verify CUDA installation
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Missing Dependencies
```bash
# Reinstall environment
conda env remove -n SEA
conda env create -f environment.yaml
```

### LLM API Errors
- Check API key is set: `echo $OPENAI_API_KEY`
- Verify internet connection
- LLM features are optional - pipeline works without them

### Low Detection Count
- Adjust `intensity_threshold` in config
- Modify `min_object_size` and `max_object_size`
- Check image quality (SNR) in inspection phase

## Model Training (Advanced)

### Noise2Void Training
The pipeline includes N2V support, but you may need to train custom models for your specific noise characteristics. See [Noise2Void documentation](https://github.com/juglab/n2v).

### StarDist Custom Models
Pre-trained StarDist models work well, but you can fine-tune on your data. See [StarDist documentation](https://github.com/stardist/stardist).

## Visualization GUI

A modern React TypeScript web interface for visualizing alignment results:

- **3-Step Workflow**: Before/After comparison → Movement analysis → Final results
- **Interactive**: Frame selection, sortable tables, image preview
- **Mock Data Mode**: Test without running the pipeline

### Quick Start

```bash
# Setup (one time)
conda activate SEA
bash setup_visualization.sh
```

### Running the GUI (2 terminals required)

**Terminal 1 — Backend API server:**
```bash
conda activate SEA
python api_server_extended.py
# Runs on http://localhost:5000
```

> **Important:** Use `api_server_extended.py`, not `api_server.py`. The extended server provides all backend endpoints required by the GUI, including image preprocessing, alignment, and exosome detection. The basic `api_server.py` only serves raw file previews and will cause 404 errors for most GUI features.

**Terminal 2 — Frontend dev server:**
```bash
cd frontend
npm run dev
# Open http://localhost:3000
```

See `QUICKSTART_VISUALIZATION.md` for detailed instructions.

### Documentation

- `QUICKSTART_VISUALIZATION.md` - Get started in 5 minutes
- `VISUALIZATION_SETUP.md` - Comprehensive setup guide
- `VISUALIZATION_README.md` - Features and architecture
- `frontend/README.md` - Frontend development details

## Citation

If you use SEA in your research, please cite:

```
SEA: Semiautomated Exosome Analysis Pipeline
AI Agent Architecture for Multi-Channel Fluorescence Imaging
```

## License

[Specify your license here]

## Contact

[Your contact information]

