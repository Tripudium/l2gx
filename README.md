
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Pylint](https://github.com/lotzma/L2GX/actions/workflows/pylint.yml/badge.svg)](https://github.com/lotzma/L2GX/actions/workflows/pylint.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

# L2GX - Local2Global Expanded

## Overview

## Documentation

Full documentation available [here](https://l2gx.readthedocs.io/en/latest/)

## Setup

**Supported Python Versions**: 3.10, 3.11, 3.12  
**Supported Operating Systems**: macOS, Linux

### Quick Start with uv

We use [`uv`](https://docs.astral.sh/uv/) for fast Python package management. Follow these steps to get started:

#### 1. Install uv

**macOS/Linux with Homebrew:**
```shell
brew install uv
```

**macOS/Linux with curl:**
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 2. Clone and Setup

```shell
# Clone the repository
git clone https://github.com/Tripudium/L2GX.git
cd L2GX

# Install dependencies and create virtual environment
uv sync --dev

# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

#### 3. Verify Installation

```shell
# Test that everything works
uv run python -c "import l2gx; print('L2GX installed successfully!')"

# Run a quick test
uv run pytest tests/ -x
```

### uv Commands Reference

```shell
# Install all dependencies (including dev)
uv sync --dev

# Install only production dependencies
uv sync

# Add a new dependency
uv add numpy

# Add a development dependency
uv add --dev pytest

# Run a command in the virtual environment
uv run python script.py
uv run pytest
uv run jupyter notebook

# Update dependencies
uv sync --upgrade

# Remove the virtual environment
rm -rf .venv
```

### Alternative Setup (Traditional pip)

If you prefer not to use uv:

```shell
git clone https://github.com/Tripudium/L2GX.git
cd L2GX
python -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -e ".[dev]"
```

## Testing and Development

### Quick Testing with uv

```shell
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Run specific test file
uv run pytest tests/test_specific.py

# Run linting
uv run ruff check
uv run ruff format

# Execute example notebooks
uv run jupyter execute examples/demo.ipynb
```

### Advanced Testing with nox

[nox](https://nox.thea.codes) simplifies Python testing across multiple Python versions. Install nox:

```shell
# Recommended: Install with uv
uv tool install nox

# Alternatives
brew install nox      # macOS
pipx install nox      # with pipx
sudo apt install nox  # debian
sudo dnf install nox  # fedora
```

Run comprehensive tests and linting:

```shell
# Run all tests and linting
nox

# List available tasks
nox --list

# Run specific task
nox -s lint
nox -s tests
nox -s notebooks
```

### Development Workflow

For active development, set up pre-commit hooks for automatic code formatting and linting:

```shell
# Install pre-commit (recommended: use uv)
uv tool install pre-commit

# Set up pre-commit hooks
pre-commit install

# Run pre-commit on all files (optional)
pre-commit run --all-files
```

This ensures code quality checks run automatically before every commit.

### Using as a Library

If you're only using L2GX as a dependency in another project:

```shell
# Install from GitHub
pip install git+https://github.com/OxfordRSE/L2GX

# Or with uv
uv add git+https://github.com/OxfordRSE/L2GX
```

### Development Tips

```shell
# Format code
uv run ruff format .

# Check code quality
uv run ruff check .
uv run pylint l2gx/

# Type checking (if mypy is installed)
uv run mypy l2gx/

# Run examples
uv run python examples/embedding_demo.py
uv run python examples/embedding_demo.py cora  # Cora-only demo
```

## License

This project is licensed under the [MIT](LICENSE) license.

## Contributors

The following people contributed to this project ([emoji key](https://allcontributors.org/docs/en/emoji-key)).


This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.
