# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

L2Gv2 (Local2Global v2) is a Python library for inferring global embeddings from local graph embeddings trained in parallel. The project focuses on graph alignment and embedding techniques, particularly for large-scale graph analysis.

## Development Commands

### Testing and Linting
- **Run all tests and linting**: `nox` (runs tests, pylint, and ruff)
- **Run only tests**: `nox -s tests`
- **Run only linting**: `nox -s lint` 
- **Run specific test file**: `uv run pytest tests/test_local2global.py`
- **Run tests with coverage**: `uv run pytest -n auto --cov`

### Jupyter Notebooks
- **Execute all example notebooks**: `nox -s notebooks`
- **Execute specific notebook**: `uv run --with jupyter jupyter execute examples/demo.ipynb`

### Package Management
- **Install dependencies**: `uv sync --all-extras --dev`
- **Install for development**: `uv sync --dev`
- **Add new dependency**: `uv add <package-name>`

### Documentation
- **Build docs**: `cd docs && make html`
- **Check unused code**: `nox -s unused-code`

## Core Architecture

### Module Structure
- **l2gv2/align/**: Graph alignment algorithms and utilities
  - `alignment.py`: Core alignment functions and error metrics
  - `geo/`: Geometric alignment methods (Procrustes, manifold alignment)
  - `l2g/`: Local-to-global alignment transformations
  - `hierarchical.py`: Hierarchical alignment strategies
  
- **l2gv2/datasets/**: Graph dataset loaders and utilities
  - `base.py`: BaseDataset class wrapping PyTorch Geometric datasets
  - `registry.py`: Dataset registry for managing available datasets
  - Individual dataset classes (cora.py, as733.py, dgraph.py)
  
- **l2gv2/embedding/**: Graph embedding models and training
  - `embedding.py`: Base EmbeddingModel class
  - `gae/`: Graph Auto-Encoder implementations (GAE/VGAE)
  - `svd/`: SVD-based embedding methods
  - `train.py`: Training utilities and procedures

- **l2gv2/graphs/**: Graph data structures and utilities
  - `graph.py`: Base Graph class with numpy backing
  - `tgraph.py`: PyTorch Geometric graph wrapper
  - `npgraph.py`: Numpy-based graph implementation

- **l2gv2/patch/**: Graph patching and partitioning
  - `patches.py`: Patch class and patch creation utilities
  - `clustering.py`: Graph clustering and partitioning algorithms
  - `sparsify.py`: Graph sparsification methods
  - `lazy.py`: Lazy evaluation for patch coordinates

### Key Dependencies
- **PyTorch & PyTorch Geometric**: Core tensor operations and graph neural networks
- **NetworkX**: Graph manipulation and algorithms
- **Raphtory**: Temporal graph analysis
- **Polars**: High-performance DataFrame operations
- **NumPy/SciPy**: Numerical computing and sparse matrices
- **scikit-learn**: Machine learning utilities
- **PyManopt**: Manifold optimization (used in geometric alignment)

### Data Flow Pattern
1. **Dataset Loading**: Use BaseDataset to load graphs from various sources
2. **Graph Processing**: Convert between different graph representations (NetworkX, PyTorch Geometric, Raphtory)
3. **Patching**: Create overlapping patches using clustering algorithms
4. **Local Embedding**: Train embeddings on individual patches using GAE/VGAE
5. **Alignment**: Align local embeddings to create global embedding using geometric methods
6. **Evaluation**: Compute alignment errors and embedding quality metrics

### Testing Strategy
- Tests located in `tests/` directory
- Uses pytest with parallel execution (`-n auto`)
- Coverage reporting enabled
- Example notebooks serve as integration tests
- Individual dataset tests in `tests/datasets/`

### Code Quality
- **Linting**: Uses both pylint and ruff for code quality
- **Formatting**: Ruff handles code formatting
- **Type Checking**: Uses pyright for static type analysis
- **Pre-commit hooks**: Available for automated formatting and linting

## Development Notes

- The project uses `uv` for fast Python package management
- Main Python version: 3.10 (supports 3.10, 3.11, 3.12)
- Many components support both CPU and GPU computation via PyTorch
- Graph datasets are cached after first load for performance
- The alignment module supports multiple error metrics (Procrustes, local Euclidean distance)