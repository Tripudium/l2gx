# Unified Embedding Framework

This document describes the updated classification and embedding experiments that now use the unified `get_embedding()` framework with `"patched"` and `"hierarchical"` embeddings.

## Overview

The experiments have been adapted to use:

- `get_embedding("patched", aligner=..., ...)` - Unified patched embedding interface
- `get_embedding("hierarchical", aligner=..., ...)` - Unified hierarchical embedding interface  
- Configuration-driven experiments for reproducibility
- Automatic parameter extraction and aligner management

## Key Changes

### 1. Classification Experiments (`scripts/classification/`)

**Before:**
- Manual patch creation with `create_patches()`
- Manual embedder creation for each patch
- Manual aligner creation and configuration
- Complex patch-by-patch embedding loop

**After:**
- Single `get_embedding("patched", aligner=aligner, ...)` call
- Aligner created once and passed to embedder
- All patch creation, embedding, and alignment handled internally
- Simplified, consistent API

### 2. Embedding Experiments (`scripts/embedding/`)

**Before:**
- Complex conditional logic for different embedding types
- Legacy configuration format with separate `patches` and `alignment` sections
- Manual coordination between embedding and alignment

**After:**
- New unified configuration format with `patched` and `aligner` sections
- Support for both legacy and new configuration formats
- Single entry point with `get_embedding()` framework

## Configuration Formats

### New Unified Patched Configuration

```yaml
# Unified patched embedding configuration
patched:
  embedding_dim: 128
  base_method: "vgae"
  num_patches: 10
  clustering_method: "metis"
  min_overlap: 10
  target_overlap: 20
  epochs: 200
  learning_rate: 0.001
  # ... other parameters

# Separate aligner configuration  
aligner:
  method: "l2g"
  randomized_method: "randomized"
  sketch_method: "rademacher"
```

### New Hierarchical Configuration

```yaml
# Unified hierarchical embedding configuration
hierarchical:
  embedding_dim: 128
  base_method: "vgae" 
  max_patch_size: 800
  min_overlap: 64
  target_overlap: 128
  epochs: 200
  learning_rate: 0.001
  # ... other parameters

# Aligner for multi-way trees (binary trees use Procrustes automatically)
aligner:
  method: "l2g"
  randomized_method: "randomized"
  sketch_method: "gaussian"
```

## Example Usage

### Direct API Usage

```python
from l2gx.embedding import get_embedding
from l2gx.align import get_aligner

# Create aligner
l2g_aligner = get_aligner("l2g")
l2g_aligner.randomized_method = "randomized" 
l2g_aligner.sketch_method = "rademacher"

# Create patched embedder
embedder = get_embedding(
    "patched",
    embedding_dim=64,
    aligner=l2g_aligner,
    num_patches=10,
    base_method="vgae",
    epochs=100
)

# Compute embedding
embedding = embedder.fit_transform(data)
```

### Configuration-Driven Experiments

```python
# Classification experiment
python scripts/classification/classification_experiment.py unified_cora_config.yaml

# Embedding experiment  
python scripts/embedding/run_embedding_config.py config/unified_patched_l2g_config.yaml
```

## Available Configuration Files

### Classification Experiments
- `scripts/classification/unified_cora_config.yaml` - Uses unified framework for all methods

### Embedding Experiments  
- `scripts/embedding/config/unified_patched_l2g_config.yaml` - Patched embedding with L2G
- `scripts/embedding/config/unified_patched_geo_config.yaml` - Patched embedding with Geo
- `scripts/embedding/config/unified_hierarchical_config.yaml` - Hierarchical embedding

### Legacy Support
- Original configuration files still supported for backward compatibility
- Experiments automatically detect configuration format and use appropriate code path

## Benefits

1. **Consistency** - Same API across all embedding methods
2. **Simplicity** - No manual patch creation or alignment coordination  
3. **Flexibility** - Easy to switch between alignment methods
4. **Reliability** - Automatic parameter extraction and validation
5. **Reproducibility** - Configuration-driven experiments
6. **Smart Defaults** - Hierarchical embeddings automatically choose appropriate alignment

## Migration Guide

### For Classification Experiments

Old method configuration:
```yaml
methods:
  l2g_rademacher:
    # Complex manual patch/align configuration
```

New method configuration:
```yaml  
methods:
  patched_l2g_rademacher:
    # Simplified configuration, same parameters
```

### For Embedding Experiments

Old configuration:
```yaml
patches:
  num_patches: 10
  # ... patch parameters
  
alignment:
  method: "l2g"  
  # ... alignment parameters
```

New configuration:
```yaml
patched:
  num_patches: 10
  # ... embedding + patch parameters

aligner:
  method: "l2g"
  # ... alignment parameters  
```

## Examples

See `scripts/embedding/unified_embedding_example.py` for comprehensive examples showing:

1. Patched embedding with L2G alignment
2. Hierarchical embedding with smart alignment
3. Patched embedding with Geo alignment  
4. Configuration-driven experiments

Run with:
```bash
cd scripts/embedding
python unified_embedding_example.py
```