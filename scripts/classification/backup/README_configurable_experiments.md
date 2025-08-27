# Configurable Dimension Sweep Classification Experiments

A flexible, configuration-driven system for comparing embedding methods across different dimensions and datasets. Now includes **Geo alignment with randomized Rademacher L2G initialization** as the fourth embedding method.

## System Overview

The configurable experiment system consists of three main components:

1. **Configuration Files** (YAML): Define experiment parameters, methods, and datasets
2. **Experiment Runner** (`configurable_dimension_sweep.py`): Executes experiments based on config
3. **Plotting Script** (`create_accuracy_plots.py`): Generates visualizations from results

## Four Embedding Methods Supported

### 1. **Full Graph VGAE**
- Applies VGAE to the entire graph
- Best accuracy but limited scalability
- Optimal for small-medium graphs (<10K nodes)

### 2. **L2G + Rademacher**
- Patch-based VGAE with L2G alignment using Rademacher sketching
- Good distributed processing capabilities
- Configurable patch count and overlap parameters

### 3. **Hierarchical + L2G** 
- Binary tree subdivision with size bounds (800 nodes)
- Procrustes alignment between patches
- Excellent scalability-performance trade-off

### 4. **Geo + Rademacher** ✨ *NEW*
- Patch-based VGAE with Geometric alignment
- **2 epochs** of geometric optimization
- **Randomized Rademacher L2G initialization**
- Combines benefits of geometric alignment with efficient initialization

## Quick Start

### 1. Run Experiment with Default Config
```bash
# Uses experiment_config.yaml (Cora dataset, 4 methods)
python configurable_dimension_sweep.py

# Or specify custom config
python configurable_dimension_sweep.py my_config.yaml
```

### 2. Generate Plots from Results
```bash
# Single experiment plots
python create_accuracy_plots.py dimension_sweep_results

# Compare multiple experiments
python create_accuracy_plots.py cora_results pubmed_results --compare
```

## Configuration Files

### Basic Structure
```yaml
experiment:
  name: "My_Experiment"
  description: "Custom experiment description"
  output_dir: "my_results"

dataset:
  name: "Cora"  # or "PubMed", "CiteSeer", etc.
  
parameters:
  dimensions: [2, 4, 8, 16, 32, 64, 128]
  n_runs: 3
  test_size: 0.2
  random_seed: 42

methods:
  # Enable/disable and configure each method
  full_graph:
    enabled: true
    base_embedding:
      epochs: 100
      learning_rate: 0.001
  
  geo_rademacher:
    enabled: true
    alignment:
      num_epochs: 2
      use_randomized_init: true
      randomized_method: "randomized"
      sketch_method: "rademacher"
```

### Pre-configured Examples

**`experiment_config.yaml`**: Full Cora experiment (4 methods, 7 dimensions)
```bash
python configurable_dimension_sweep.py experiment_config.yaml
```

**`pubmed_experiment_config.yaml`**: PubMed dataset with optimized settings
```bash
python configurable_dimension_sweep.py pubmed_experiment_config.yaml
```

**`test_config.yaml`**: Quick validation test (2 methods, 2 dimensions)
```bash
python configurable_dimension_sweep.py test_config.yaml
```

## Advanced Usage

### Custom Dataset Configuration
```yaml
dataset:
  name: "CiteSeer"
  normalize_features: true
  data_root: "/path/to/custom/data"
```

### Method-Specific Tuning
```yaml
methods:
  geo_rademacher:
    enabled: true
    patches:
      num_patches: 15  # Custom patch count
      min_overlap: 128
      target_overlap: 256
    alignment:
      geo_method: "orthogonal"  # or "similarity", "affine"
      num_epochs: 3  # More optimization epochs
      learning_rate: 0.02
      use_scale: true
      use_randomized_init: true
      randomized_method: "randomized"
      sketch_method: "rademacher"
```

### Plotting Customization
```yaml
plotting:
  save_formats: ["pdf", "png", "svg"]
  figure_size: [20, 15]
  dpi: 300
  colors: ["#custom", "#colors"]
```

## Output Structure

Each experiment creates a results directory with:

```
results_directory/
├── raw_results.csv           # Detailed data (all runs)
├── summary_results.csv       # Aggregated statistics  
├── experiment_report.txt     # Human-readable summary
├── experiment_name_accuracy_plots.pdf  # Visualization
└── experiment_name_accuracy_plots.png
```

### Results Data Format

**Summary Results CSV**:
| method | embedding_dim | accuracy_mean | accuracy_std | embedding_time_mean |
|--------|---------------|---------------|--------------|-------------------|
| full_graph | 64 | 0.8684 | 0.0083 | 1.9725 |
| geo_rademacher | 64 | 0.7856 | 0.0234 | 4.2341 |

## Performance Comparison

Based on Cora dataset experiments:

| Method | Best Accuracy | Optimal Dim | Avg Time (128D) | Scalability |
|--------|---------------|-------------|-----------------|-------------|
| **Full Graph** | 86.96% | 128D | ~4s | Limited |
| **Hierarchical + L2G** | 83.27% | 128D | ~9s | Excellent |
| **Geo + Rademacher** | ~78-82%* | 64-128D | ~8s | Good |
| **L2G + Rademacher** | 72.69% | 64D | ~10s | Good |

*Exact performance depends on geometric optimization convergence

## Key Features

### 1. **Dataset Flexibility**
- Support for any PyTorch Geometric dataset
- Configurable preprocessing options
- Easy addition of new datasets

### 2. **Method Modularity**
- Enable/disable methods independently
- Fine-tune parameters per method
- Easy addition of new embedding approaches

### 3. **Experiment Reproducibility**
- Fixed random seeds
- Complete configuration logging
- Detailed experimental reports

### 4. **Scalable Analysis**
- Multiple runs with statistical analysis
- Configurable dimensions and parameters
- Batch processing capabilities

### 5. **Visualization Tools**
- Automatic plot generation
- Multi-experiment comparisons
- Publication-ready figures (PDF)

## Best Practices

### For Small Graphs (<5K nodes):
```yaml
methods:
  full_graph:
    enabled: true
    base_embedding:
      epochs: 100
```

### For Medium Graphs (5K-50K nodes):
```yaml
methods:
  hierarchical_l2g:
    enabled: true
    hierarchical:
      max_patch_size: 800
  geo_rademacher:
    enabled: true
    alignment:
      num_epochs: 2
```

### For Large Graphs (>50K nodes):
```yaml
methods:
  l2g_rademacher:
    enabled: true
    patches:
      num_patches: 50  # Increase for larger graphs
  geo_rademacher:
    enabled: true
    patches:
      num_patches: 50
```

## Troubleshooting

### Common Issues

**1. Memory Errors with Full Graph**
```yaml
methods:
  full_graph:
    enabled: false  # Disable for large graphs
```

**2. Slow Geo Convergence**
```yaml
methods:
  geo_rademacher:
    alignment:
      num_epochs: 1  # Reduce epochs
      learning_rate: 0.02  # Increase learning rate
```

**3. Poor L2G Performance**
```yaml
methods:
  l2g_rademacher:
    patches:
      num_patches: 20  # Increase patches
      target_overlap: 1024  # Increase overlap
```

## Extension Guide

### Adding New Datasets
1. Ensure dataset is available in PyTorch Geometric
2. Add dataset name to configuration
3. Adjust preprocessing parameters if needed

### Adding New Methods
1. Implement method in `configurable_dimension_sweep.py`
2. Add method configuration schema
3. Update plotting display names

### Custom Metrics
1. Modify `run_classification()` method
2. Add metric computation and storage
3. Update plotting scripts for new metrics

This configurable system provides a comprehensive framework for graph embedding evaluation, making it easy to conduct systematic experiments across different datasets, methods, and parameters while maintaining reproducibility and generating publication-quality results.