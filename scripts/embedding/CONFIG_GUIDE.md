# Embedding Configuration Guide

This guide explains how to use YAML configuration files to define complete embedding experiments with different base methods, patching strategies, and alignment procedures.

## Configuration Structure

Each embedding process is defined by:
1. **Base embedding method** (VGAE, GAE, SVD, DGI, GraphSAGE)
2. **Patching specification** (number of patches, clustering algorithm, parameters)
3. **Alignment procedure** (L2G, Geo, with various parameters)
4. **Optional hierarchical alignment** (recursive patch decomposition)

## Configuration Files

### Available Templates

- `embedding_config_l2g.yaml` - L2G alignment with randomized Rademacher
- `embedding_config_geo.yaml` - Geometric alignment with 2 epochs and randomized initialization  
- `embedding_config_hierarchical.yaml` - Hierarchical patching example

## Configuration Sections

### 1. Experiment Settings
```yaml
experiment:
  name: "My_Experiment"
  description: "Description of the experiment"
  output_dir: "results/my_experiment"
```

### 2. Dataset Configuration
```yaml
dataset:
  name: "Cora"  # Options: Cora, PubMed, CiteSeer, etc.
  normalize_features: false
  # data_root: "/custom/path"  # Optional custom data location
```

### 3. Base Embedding Method
```yaml
embedding:
  method: "vgae"  # Options: vgae, gae, svd, dgi, graphsage
  embedding_dim: 128
  hidden_dim_multiplier: 2  # hidden_dim = embedding_dim * multiplier
  epochs: 10000
  learning_rate: 0.001
  patience: 20
  verbose: false
```

### 4. Patch Generation
```yaml
patches:
  # Standard patching
  num_patches: 10  # Set to 1 for whole graph embedding
  clustering_method: "metis"  # Options: metis, fennel, louvain
  min_overlap: 256
  target_overlap: 512
  sparsify_method: "resistance"  # Options: resistance, edge_sampling, nearest_neighbor
  target_patch_degree: 4
  use_conductance_weighting: true
  verbose: false
  
  # Hierarchical patching (optional)
  hierarchical:
    enabled: false  # Set to true for hierarchical approach
    max_patch_size: 500  # Split patches larger than this
    k_patches: 3  # Number of child patches per subdivision
    min_patch_size: 50  # Stop subdivision at this size
```

### 5. Alignment Configuration

#### L2G Alignment
```yaml
alignment:
  method: "l2g"
  randomized_method: "randomized"  # Options: "randomized", "standard"
  sketch_method: "rademacher"  # Options: "rademacher", "gaussian", "sparse"
  scale: false
  verbose: true
```

#### Geometric Alignment
```yaml
alignment:
  method: "geo"
  geo_method: "orthogonal"  # Options: "orthogonal", "similarity", "affine"
  num_epochs: 2
  learning_rate: 0.01
  use_scale: true
  use_randomized_init: true
  randomized_method: "sparse_aware"  # Options: "sparse_aware", "rademacher", "gaussian"
  sketch_method: "rademacher"
  verbose: true
```

### 6. Output Configuration
```yaml
output:
  save_embeddings: true
  save_patches: false
  save_alignment_matrices: false
  format: "npz"  # Options: npz, pkl, h5
```

## Usage Examples

### Run L2G Experiment
```bash
python run_embedding_config.py embedding_config_l2g.yaml
```

### Run Geo Experiment  
```bash
python run_embedding_config.py embedding_config_geo.yaml
```

### Run Hierarchical Experiment
```bash
python run_embedding_config.py embedding_config_hierarchical.yaml
```

## Configuration Examples

### L2G with Randomized Rademacher
```yaml
alignment:
  method: "l2g"
  randomized_method: "randomized"
  sketch_method: "rademacher"
  scale: false
  verbose: true
```

### Geo with 2 Epochs and Randomized Initialization
```yaml
alignment:
  method: "geo" 
  geo_method: "orthogonal"
  num_epochs: 2
  learning_rate: 0.01
  use_scale: true
  use_randomized_init: true
  randomized_method: "sparse_aware"
  sketch_method: "rademacher"
  verbose: true
```

### Hierarchical Patching
```yaml
patches:
  hierarchical:
    enabled: true
    max_patch_size: 1000  # Split large patches
    k_patches: 4  # Create 4 children per split
    min_patch_size: 200  # Minimum leaf size
    max_depth: 3  # Maximum recursion depth
```

## Base Embedding Methods

### VGAE (Variational Graph Auto-Encoder)
```yaml
embedding:
  method: "vgae"
  embedding_dim: 128
  hidden_dim_multiplier: 2
  epochs: 10000
  learning_rate: 0.001
```

### GAE (Graph Auto-Encoder)
```yaml
embedding:
  method: "gae"
  embedding_dim: 128
  hidden_dim_multiplier: 2
  epochs: 5000
  learning_rate: 0.001
```

### SVD (Singular Value Decomposition)
```yaml
embedding:
  method: "svd"
  embedding_dim: 128
  # SVD has fewer parameters
```

### DGI (Deep Graph Infomax)
```yaml
embedding:
  method: "dgi"
  embedding_dim: 128
  hidden_dim_multiplier: 2
  epochs: 8000
  learning_rate: 0.001
```

### GraphSAGE
```yaml
embedding:
  method: "graphsage"
  embedding_dim: 128
  hidden_dim_multiplier: 2
  epochs: 5000
  learning_rate: 0.01
```

## Clustering Methods

- **metis**: METIS graph partitioning (recommended for most cases)
- **fennel**: Fennel streaming clustering
- **louvain**: Louvain community detection
- **hierarchical**: Hierarchical clustering

## Sparsification Methods

- **resistance**: Effective resistance-based sparsification (recommended)
- **edge_sampling**: Random edge sampling
- **nearest_neighbor**: k-nearest neighbor graphs

## Output Files

The experiment generates:
- `embedding_results.npz` - Main embedding results
- `experiment_metadata.yaml` - Experiment configuration and metadata
- Optional patch and alignment data (if enabled)

## Advanced Parameters

### L2G Advanced Options
```yaml
alignment:
  method: "l2g"
  # Advanced parameters (uncomment to use)
  # sketch_dimension: null  # Auto-determined
  # regularization: 1e-6
  # max_iterations: 1000
  # tolerance: 1e-8
```

### Geo Advanced Options
```yaml
alignment:
  method: "geo"
  # Advanced parameters (uncomment to use)
  # manifold_type: "stiefel"
  # optimizer: "riemannian_sgd"  
  # momentum: 0.9
  # weight_decay: 1e-4
  # gradient_clipping: 1.0
```

## Tips and Best Practices

1. **Start with templates**: Use provided templates and modify as needed
2. **Embedding dimensions**: 64-128 works well for most datasets
3. **Patch overlap**: 256-512 overlap provides good connectivity
4. **L2G vs Geo**: L2G is faster, Geo can be more accurate with proper tuning
5. **Hierarchical**: Use for very large graphs (>50k nodes)
6. **Epochs**: VGAE needs more epochs (10k+), GAE and others can use fewer
7. **Randomized methods**: Usually provide good speed/quality tradeoff

## Troubleshooting

- **Out of memory**: Reduce embedding dimensions or patch sizes
- **Poor alignment**: Try different initialization methods or increase epochs
- **Slow performance**: Use randomized methods or reduce epochs
- **Config errors**: Check YAML syntax and required sections