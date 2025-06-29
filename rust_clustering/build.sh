#!/bin/bash

# Build script for L2G Clustering Rust extension

set -e

echo "=" * 60
echo "L2G Clustering Rust Extension Build Script"
echo "=" * 60

# Function to check command availability
check_command() {
    if ! command -v $1 &> /dev/null; then
        return 1
    fi
    return 0
}

# Function to install maturin
install_maturin() {
    echo "Installing maturin..."
    if check_command pip; then
        pip install maturin
    elif check_command pip3; then
        pip3 install maturin
    else
        echo "ERROR: pip not found. Please install pip first."
        exit 1
    fi
}

# Function to install numpy if missing
install_numpy() {
    echo "Checking for numpy..."
    python -c "import numpy" 2>/dev/null || {
        echo "Installing numpy..."
        pip install "numpy>=1.20.0"
    }
}

# Check if Rust is installed
echo "Checking Rust installation..."
if ! check_command cargo || ! check_command rustc; then
    echo "ERROR: Rust is not installed."
    echo ""
    echo "Please install Rust first:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "  source ~/.cargo/env"
    echo ""
    echo "Or visit: https://rustup.rs/"
    exit 1
fi

# Display Rust version
RUST_VERSION=$(rustc --version)
echo "Found Rust: $RUST_VERSION"

# Check if maturin is installed
echo "Checking maturin installation..."
if ! check_command maturin; then
    install_maturin
fi

# Display maturin version
MATURIN_VERSION=$(maturin --version)
echo "Found maturin: $MATURIN_VERSION"

# Install numpy if needed
install_numpy

# Check Python version
PYTHON_VERSION=$(python --version 2>&1)
echo "Using Python: $PYTHON_VERSION"

# Clean previous builds
echo "Cleaning previous builds..."
if [ -d "target" ]; then
    cargo clean
fi

# Set optimization flags for maximum performance
echo "Setting optimization flags..."
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"

# Build the extension
echo "Building Rust extension in release mode..."
echo "This may take a few minutes for the first build..."

if maturin develop --release; then
    echo ""
    echo "✅ Build completed successfully!"
    echo ""
    
    # Test the installation
    echo "Testing installation..."
    if python -c "import l2g_clustering; print('✅ Rust extension loaded successfully!')" 2>/dev/null; then
        echo ""
        echo "🚀 Installation verified! The Rust clustering extension is ready to use."
        echo ""
        echo "Quick test:"
        echo "  python -c \"from l2gv2.patch.clustering import is_rust_available; print(f'Rust available: {is_rust_available()}')\""
        echo ""
        echo "Benchmarking:"
        echo "  python ../../scripts/benchmark_clustering.py --nodes 1000 --clusters 10"
    else
        echo "⚠️  Build succeeded but import failed. You may need to:"
        echo "   1. Restart your Python session"
        echo "   2. Check PYTHONPATH includes the L2G project"
        echo "   3. Verify you're using the same Python version that was used for building"
    fi
else
    echo ""
    echo "❌ Build failed!"
    echo ""
    echo "Common solutions:"
    echo "  1. Make sure you have the required build tools:"
    echo "     - Linux: sudo apt-get install build-essential python3-dev"
    echo "     - macOS: xcode-select --install"
    echo "     - Windows: Install Visual Studio Build Tools"
    echo ""
    echo "  2. Try rebuilding with a specific Python interpreter:"
    echo "     maturin develop --release --interpreter $(which python)"
    echo ""
    echo "  3. Check the full error message above for specific issues"
    exit 1
fi