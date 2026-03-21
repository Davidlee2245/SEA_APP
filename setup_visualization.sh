#!/bin/bash
# Setup script for SEA Visualization GUI

set -e

echo "===================================="
echo "SEA Visualization Setup"
echo "===================================="
echo ""

# Check if conda environment is active
if [[ "$CONDA_DEFAULT_ENV" != "SEA" ]]; then
    echo "⚠️  Warning: SEA conda environment is not active"
    echo "Please run: conda activate SEA"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install Python dependencies for API server
echo "📦 Installing Python API dependencies..."
pip install flask flask-cors

echo ""
echo "✅ Python dependencies installed"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed"
    echo ""
    echo "Please install Node.js 18+ using one of these methods:"
    echo "  1. Conda: conda install -c conda-forge nodejs"
    echo "  2. Download: https://nodejs.org/"
    echo ""
    exit 1
fi

NODE_VERSION=$(node --version)
echo "✅ Node.js $NODE_VERSION detected"
echo ""

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd frontend

if [ ! -f package.json ]; then
    echo "❌ package.json not found in frontend directory"
    exit 1
fi

npm install

echo ""
echo "✅ Frontend dependencies installed"
echo ""

cd ..

# Create sample directories if they don't exist
echo "📁 Checking data directories..."
mkdir -p data/input
mkdir -p data/output

echo "✅ Directory structure ready"
echo ""

echo "===================================="
echo "✅ Setup Complete!"
echo "===================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Run the SEA pipeline to generate data:"
echo "   python pipeline.py"
echo ""
echo "2. Start the visualization:"
echo ""
echo "   Option A - Development with Mock Data:"
echo "   cd frontend && npm run dev"
echo "   (Then open http://localhost:3000)"
echo ""
echo "   Option B - Full Stack with Real Data:"
echo "   Terminal 1: python api_server.py"
echo "   Terminal 2: cd frontend && npm run dev"
echo "   (Uncheck 'Use Mock Data' in the UI)"
echo ""
echo "📖 For more details, see VISUALIZATION_SETUP.md"
echo ""


