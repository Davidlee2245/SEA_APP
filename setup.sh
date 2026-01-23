#!/bin/bash
# Setup script for SEA Exosome Analysis Pipeline

set -e

echo "=== SEA Pipeline Setup ==="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^SEA "; then
    echo "Warning: Conda environment 'SEA' already exists"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n SEA -y
    else
        echo "Skipping environment creation. Activate with: conda activate SEA"
        exit 0
    fi
fi

# Create environment
echo "Creating Conda environment 'SEA'..."
conda env create -f environment.yaml

echo ""
echo "=== Installing N2V (Optional) ==="
echo "N2V requires a different csbdeep version than StarDist."
echo "Installing N2V separately..."
conda activate SEA
pip install n2v --no-deps || echo "Warning: N2V installation failed. You can install it later manually."

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment, run:"
echo "  conda activate SEA"
echo ""
echo "To verify installation, run:"
echo "  python -c \"import torch; print('CUDA:', torch.cuda.is_available())\""
echo "  python -c \"import kornia; import stardist; print('Dependencies OK')\""
echo ""
echo "To run the pipeline:"
echo "  python pipeline.py"

