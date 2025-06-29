# Setting Up Rust Clustering Extensions for L2GX

This guide provides complete instructions for building and using the high-performance Rust clustering implementations in the L2GX framework.

## Quick Start

```bash
# 1. Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 2. Install Python dependencies
pip install maturin numpy

# 3. Build the extension
cd rust_clustering
chmod +x build.sh
./build.sh

# 4. Test the installation
python -c "from l2gx.patch.clustering import is_rust_available; print(f'Rust available: {is_rust_available()}')"
```

## Detailed Installation Guide

### Prerequisites

1. **Rust Programming Language**
   - Install from [rustup.rs](https://rustup.rs/)
   - Minimum version: 1.70+
   - Includes `cargo` (Rust package manager)

2. **Python Development Environment**
   - Python 3.8+
   - Development headers (python3-dev on Linux)
   - pip package manager

3. **Build Tools**
   - **Linux**: `build-essential` package
   - **macOS**: Xcode command line tools
   - **Windows**: Visual Studio Build Tools

### Step 1: Install Rust

#### On Linux/macOS:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

#### On Windows:
1. Download and run [rustup-init.exe](https://rustup.rs/)
2. Follow the installation wizard
3. Restart your terminal

#### Verify Installation:
```bash
rustc --version  # Should show: rustc 1.70+ 
cargo --version  # Should show: cargo 1.70+
```

### Step 2: Install Python Dependencies

```bash
# Install maturin (builds Rust extensions for Python)
pip install maturin

# Install required Python packages
pip install numpy>=1.20.0

# Optional: Install development tools
pip install pytest matplotlib seaborn  # For benchmarking
```

### Step 3: Install Platform-Specific Build Tools

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev
```

#### Linux (CentOS/RHEL):
```bash
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel
```

#### macOS:
```bash
# Install Xcode command line tools
xcode-select --install

# For Apple Silicon Macs, you may need:
export MACOSX_DEPLOYMENT_TARGET=10.9
```

#### Windows:
1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Select "C++ build tools" workload
3. Restart your computer

### Step 4: Build the Rust Extension

#### Option A: Automated Build (Recommended)
```bash
cd /path/to/L2GX/rust_clustering
chmod +x build.sh
./build.sh
```

#### Option B: Manual Build
```bash
cd rust_clustering

# Development build (faster compilation, slower runtime)
maturin develop

# Release build (slower compilation, much faster runtime - RECOMMENDED)
maturin develop --release
```

#### Option C: Build Wheel for Distribution
```bash
cd rust_clustering
maturin build --release
pip install target/wheels/l2g_clustering-*.whl
```

### Step 5: Verify Installation

#### Test Rust Extension Loading:
```bash
python -c "import l2g_clustering; print('Rust extension loaded successfully!')"
```

#### Test L2G Integration:
```python
from l2gx.patch.clustering import is_rust_available, get_rust_info

print(f"Rust available: {is_rust_available()}")
if is_rust_available():
    print("Rust info:", get_rust_info())
```

#### Run a Simple Benchmark:
```python
from l2gx.graphs import TGraph
from l2gx.patch.clustering import fennel_clustering_rust
import torch

# Create a small test graph
edge_index = torch.randint(0, 100, (2, 200))
graph = TGraph(edge_index, num_nodes=100)

# Test Rust clustering
clusters = fennel_clustering_rust(graph, num_clusters=10, verbose=True)
print(f"Clustering completed: {len(torch.unique(clusters))} clusters found")
```

## Troubleshooting

### Common Error Messages and Solutions

#### 1. "Rust compiler not found"
```bash
error: Microsoft Visual Studio C++ Build tools is required (Windows)
error: linker `cc` not found (Linux)
error: xcrun: error: invalid active developer path (macOS)
```

**Solutions:**
- **Windows**: Install Visual Studio Build Tools
- **Linux**: `sudo apt-get install build-essential`
- **macOS**: `xcode-select --install`

#### 2. "Python.h not found"
```bash
error: Python.h: No such file or directory
fatal error: 'Python.h' file not found
```

**Solutions:**
- **Ubuntu/Debian**: `sudo apt-get install python3-dev`
- **CentOS/RHEL**: `sudo yum install python3-devel`
- **macOS**: Reinstall Python with homebrew: `brew install python`
- **Windows**: Use official Python installer (includes headers)

#### 3. "maturin not found"
```bash
command not found: maturin
```

**Solution:**
```bash
pip install --upgrade pip
pip install maturin
```

#### 4. "ImportError: dynamic module does not define module export function"
This usually means Python version mismatch between build and runtime.

**Solutions:**
```bash
# Rebuild with explicit Python interpreter
maturin develop --release --interpreter $(which python)

# Or specify exact version
maturin develop --release --interpreter python3.9
```

#### 5. Permission denied errors
```bash
# Make build script executable
chmod +x build.sh

# Check write permissions in target directory
ls -la target/
```

### Performance Optimization

#### For Maximum Performance:
```bash
# Use native CPU optimizations
export RUSTFLAGS="-C target-cpu=native"
maturin develop --release

# Enable all optimizations (longer compile time)
export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=fat"
maturin develop --release
```

#### For Development/Debugging:
```bash
# Debug build with symbols
maturin develop --debug

# Enable Rust backtraces
export RUST_BACKTRACE=1
python your_script.py
```

## Usage Examples

### Basic Usage
```python
from l2gx.patch.clustering import fennel_clustering_rust
from l2gx.graphs import TGraph

# Load your graph
graph = TGraph(edge_index, num_nodes=10000)

# Fast Rust clustering
clusters = fennel_clustering_rust(graph, num_clusters=100)
```

### Parallel Processing
```python
# For large graphs, use parallel implementation
clusters = fennel_clustering_rust(
    graph, 
    num_clusters=100, 
    parallel=True,  # Enable multi-threading
    verbose=True
)
```

### Benchmarking
```python
from l2gx.patch.clustering import benchmark_rust_vs_python

# Compare Rust vs Python performance
results = benchmark_rust_vs_python(graph, num_clusters=50, num_runs=5)
print(f"Speedup: {results['speedup']:.2f}x")
print(f"Rust time: {results['rust_mean_time']:.3f}s")
print(f"Python time: {results['python_mean_time']:.3f}s")
```

### Full Benchmarking Suite
```bash
# Run comprehensive benchmarks
cd /path/to/L2GX
python scripts/benchmark_clustering.py

# Test specific graph size
python scripts/benchmark_clustering.py --nodes 5000 --clusters 50 --runs 3
```

## Integration with L2G Workflow

Once installed, the Rust implementations are automatically available in the L2G clustering registry:

```python
from l2gx.patch.clustering import get_clustering_algorithm

# Get Rust implementation
rust_fennel = get_clustering_algorithm('fennel_rust')

# Use in L2G pipeline
clusters = rust_fennel(graph, num_clusters=100)
```

## Updating and Maintenance

### Updating Rust
```bash
rustup update
```

### Rebuilding After Updates
```bash
cd rust_clustering
cargo clean
maturin develop --release
```

### Uninstalling
```bash
pip uninstall l2g-clustering
```

## Getting Help

If you encounter issues:

1. **Check the error message** against common issues above
2. **Verify prerequisites** are correctly installed
3. **Try a clean rebuild**: `cargo clean && maturin develop --release`
4. **Check Python/Rust versions** are compatible
5. **Enable debug output**: `export RUST_BACKTRACE=1`

For additional support, check the project documentation or create an issue with:
- Your operating system and version
- Python version (`python --version`)
- Rust version (`rustc --version`)
- Complete error message
- Steps to reproduce the issue