# L2GX Embedding & Visualization System

A modular, configuration-driven system for generating and visualizing graph embeddings using Local2Global (L2G) methods.

## Quick Start

```bash
# Run patched L2G embedding with visualization
python plot_embeddings.py configs/cora_patched.yaml

# Run whole graph embedding with visualization  
python plot_embeddings.py configs/cora_whole.yaml

# Run just the embedding (no visualization)
python embedding_experiment.py configs/cora_patched.yaml
```

## Architecture

The system is split into two main components:

### 1. Core Embedding (`embedding_experiment.py`)
- **Purpose**: Handles dataset loading, embedding computation, and result saving
- **Input**: YAML configuration file
- **Output**: Embedding arrays (.npy) and metadata (YAML/txt)
- **Key Feature**: `num_patches=1` triggers whole-graph embedding, `num_patches>1` triggers patched L2G

### 2. Visualization (`plot_embeddings.py`) 
- **Purpose**: Uses `EmbeddingExperiment` to compute embeddings, then creates visualizations
- **Input**: YAML configuration file
- **Output**: UMAP plots, individual patch plots, grid overviews
- **Key Feature**: Automatically handles visualization based on embedding type

## Configuration Format

Clean YAML configs with **no visualization parameters** (handled automatically):

```yaml
experiment:
  name: "experiment_name"
  output_dir: "results/folder"

dataset:
  name: "Cora"

embedding:
  method: "vgae"
  embedding_dim: 128
  hidden_dim_multiplier: 2
  epochs: 1000
  learning_rate: 0.001
  patience: 20

patches:
  num_patches: 10  # Use 1 for whole graph
  clustering_method: "metis"
  # ... other patch parameters

alignment:
  method: "l2g"
  scale: false
```

## Available Configurations

### Pre-made Configs
- **`configs/cora_patched.yaml`** - Cora with 10-patch L2G embedding
- **`configs/cora_whole.yaml`** - Cora with whole graph embedding (`num_patches=1`)
- **`configs/pubmed_patched.yaml`** - PubMed with optimized patch parameters
- **`configs/dgi_patched.yaml`** - Cora with DGI embedding method

### Supported Datasets
- `Cora` - Citation network (2708 nodes, 7 classes)
- `PubMed` - Citation network (19717 nodes, 3 classes) 
- `CiteSeer` - Citation network
- Any dataset in L2GX registry

### Embedding Methods
- `vgae` - Variational Graph Auto-Encoder
- `gae` - Graph Auto-Encoder
- `dgi` - Deep Graph Infomax
- `graphsage` - GraphSAGE

## Key Features

### 🔧 **Unified Embedding Logic**
- **Whole Graph**: Set `num_patches: 1` in config
- **Patched L2G**: Set `num_patches: >1` in config
- Same codebase handles both approaches seamlessly

### 📊 **Automatic Visualization**
- **Main embedding plot**: Named based on dataset and type (e.g., `7classes_patched_l2g.png`)
- **Individual patch plots**: Only created for patched embeddings
- **Grid overview**: Shows all patches before alignment
- **Consistent styling**: Datashader-based old-style plots

### 💾 **Comprehensive Output**
Each experiment generates:
```
results/experiment_name/
├── embedding.npy              # Main embedding array
├── 7classes_patched_l2g.png   # Main visualization  
├── experiment_results.yaml    # Complete metadata
├── summary.txt                # Human-readable summary
└── individual_patches/        # (if patches used)
    ├── patch_01_embedding.png
    ├── ...
    └── patch_grid_overview.png
```

### 🏗️ **Modular Design**
- **`EmbeddingExperiment`**: Core logic, no visualization dependencies
- **`EmbeddingVisualizer`**: Visualization logic, uses completed experiment
- **Clean separation**: Can run embeddings without visualization
- **Reusable**: Easy to extend with new visualization types

## Usage Examples

### Compare Approaches
```bash
# Generate patched L2G embedding
python plot_embeddings.py configs/cora_patched.yaml

# Generate whole graph embedding  
python plot_embeddings.py configs/cora_whole.yaml

# Compare results in results/cora_patched/ vs results/cora_whole/
```

### Large Dataset
```bash
# PubMed with optimized parameters
python plot_embeddings.py configs/pubmed_patched.yaml
```

### Different Methods
```bash
# DGI method instead of VGAE
python plot_embeddings.py configs/dgi_patched.yaml
```

### Embedding Only (No Plots)
```bash
# Just compute embedding, save arrays
python embedding_experiment.py configs/cora_patched.yaml
```

## Implementation Details

### Fixed TGraph.subgraph() Bug
The system includes a critical fix to `l2gx/graphs/tgraph.py` that resolves edge_index ordering issues:
- **Problem**: Old implementation created spurious self-loops and dropped valid edges
- **Fix**: Proper edge filtering where both endpoints are in subgraph node set
- **Impact**: L2G embeddings now show proper class separation

### Smart Patch Detection
- `num_patches=1`: Automatically uses whole graph embedding (no patches created)
- `num_patches>1`: Creates patches and applies L2G alignment
- Visualization automatically adapts based on embedding type

### Performance Optimizations
- **Parallel patch generation**: Uses efficient clustering algorithms
- **Optimized visualization**: Different DPI settings for different plot types
- **Memory efficient**: Saves embeddings immediately, manages large datasets

## Configuration Tips

### For Different Datasets
- **Small datasets (Cora)**: 5-15 patches, standard parameters
- **Large datasets (PubMed)**: 15-25 patches, larger overlaps, more epochs
- **Very large**: Consider higher `target_patch_degree` and more patches

### For Different Methods
- **VGAE/GAE**: `hidden_dim_multiplier: 2-4` works well
- **DGI**: Often good with `hidden_dim_multiplier: 1`
- **GraphSAGE**: May need different learning rates

### Performance vs Quality
- **More patches**: Better scalability, potentially more alignment challenges
- **Larger overlaps**: Better alignment, higher memory usage
- **Higher degrees**: Denser patches, more compute per patch

## Error Handling

Common issues and solutions:
- **Missing dataset**: Check dataset name spelling in config
- **Memory errors**: Reduce `num_patches` or `embedding_dim`
- **Convergence issues**: Increase `patience` or adjust `learning_rate`
- **Poor class separation**: Check `use_conductance_weighting: true`

## Extending the System

### Adding New Datasets
1. Ensure dataset is available in L2GX registry
2. Create config file with appropriate parameters
3. Test with `embedding_experiment.py` first

### Adding New Visualization Types
1. Extend `EmbeddingVisualizer` class
2. Add new methods for your visualization type
3. Update `create_all_visualizations()` to include them

### Adding New Embedding Methods
1. Implement in L2GX embedding registry
2. Add method name to config
3. Tune `hidden_dim_multiplier` and other parameters

This modular design makes the system highly extensible while maintaining clean separation of concerns between embedding computation and visualization.