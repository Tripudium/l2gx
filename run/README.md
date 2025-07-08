# L2GX Experiments Configuration System

This directory contains a comprehensive configuration management system for running experiments with the L2GX framework on the Cora dataset.

## Overview

The configuration system provides:

- **YAML-based configuration files** for easy parameter management
- **Type-safe configuration classes** with validation
- **Experiment orchestration** for node reconstruction and classification tasks
- **Hyperparameter search capabilities** (grid search, random search, Optuna)
- **Reproducible experiments** with seed management
- **Comprehensive logging and visualization**

## Files Structure

```
experiments/
├── config.yaml              # Full configuration with all parameters
├── config_simple.yaml       # Simplified configuration for quick testing
├── config_manager.py        # Configuration management system
├── cora_experiments.py      # Main experiment runner
├── README.md                # This file
├── results/                 # Experiment outputs (created automatically)
├── logs/                    # Log files (created automatically)
└── plots/                   # Generated visualizations (created automatically)
```

## Quick Start

### 1. Run a Simple Test

```bash
# Run with simplified configuration (faster)
cd experiments
python cora_experiments.py --config config_simple.yaml
```

### 2. Run Full Experiments

```bash
# Run with full configuration
python cora_experiments.py --config config.yaml
```

### 3. Custom Output Directory

```bash
python cora_experiments.py --config config.yaml --output-dir my_custom_results
```

## Configuration Sections

### Dataset Configuration
```yaml
dataset:
  name: "Cora"                    # Dataset to use
  use_default_splits: true        # Use predefined train/val/test splits
  train_ratio: 0.6               # Custom split ratios (if use_default_splits: false)
  val_ratio: 0.2
  test_ratio: 0.2
  split_seed: 42                 # Seed for reproducible splits
```

### Embedding Configuration
```yaml
embedding:
  method: "vgae"                 # Embedding method: ['gae', 'vgae', 'svd', 'graphsage', 'dgi']
  embedding_dim: 64              # Embedding dimension
  hidden_dim: 32                 # Hidden layer dimension (neural methods)
  num_epochs: 200                # Training epochs (neural methods)
  learning_rate: 0.01            # Learning rate
  dropout: 0.5                   # Dropout rate
  graphsage_aggregator: "mean"   # GraphSAGE aggregator: ['mean', 'max', 'lstm']
  dgi_encoder: "gcn"             # DGI encoder: ['gcn', 'gat', 'sage']
  svd_matrix_type: "normalized"  # SVD matrix: ['adjacency', 'laplacian', 'normalized']
```

### Patch Configuration
```yaml
patches:
  num_patches: 10                # Number of patches (set to 1 for global embedding)
  clustering_method: "metis"     # Clustering: ['metis', 'louvain', 'fennel', 'hierarchical']
  min_overlap: 27                # Minimum overlap between patches
  target_overlap: 54             # Target overlap
  sparsification_method: "resistance"  # Sparsification method
```

### Task Configurations
```yaml
node_reconstruction:
  loss_function: "mse"           # Loss function
  evaluation_metrics: ["mse", "mae", "cosine_similarity"]
  reconstruction_method: "autoencoder"

node_classification:
  classifier: "logistic_regression"  # Classifier type
  evaluation_metrics: ["accuracy", "f1_macro", "f1_micro", "precision", "recall"]
  stratify: true                 # Use stratified sampling
```

## Experimental Tasks

### Task 1: Node Reconstruction
- **Objective**: Reconstruct node features from learned embeddings
- **Methods**: Linear decoder, MLP decoder, autoencoder
- **Metrics**: MSE, MAE, cosine similarity
- **Use case**: Evaluate how well embeddings preserve node information

### Task 2: Node Classification
- **Objective**: Classify nodes into their respective categories
- **Methods**: Logistic regression, SVM, Random Forest, MLP
- **Metrics**: Accuracy, F1-score (macro/micro), precision, recall
- **Use case**: Evaluate embedding quality for downstream tasks

## Hyperparameter Search

Enable hyperparameter search in your configuration:

```yaml
hyperparameter_search:
  enabled: true
  method: "grid"                 # 'grid', 'random', or 'optuna'
  n_trials: 50                   # For random/optuna search
  search_space:
    embedding_dim: [32, 64, 128]
    hidden_dim: [16, 32, 64]
    learning_rate: [0.001, 0.01, 0.1]
    num_epochs: [100, 200, 300]
    num_patches: [5, 10, 15, 20]
```

## Output Structure

After running experiments, you'll find:

```
results/
├── final_results.json          # Aggregated results (JSON format)
├── final_results.pkl           # Aggregated results (pickle format)
├── intermediate/               # Individual run results (if enabled)
│   ├── run_000_results.json
│   ├── run_001_results.json
│   └── ...
└── plots/
    └── results_summary.png     # Visualization of results
```

## Example Usage Patterns

### Quick Development Testing
```bash
# Use simple config for rapid iteration
python cora_experiments.py --config config_simple.yaml
```

### Production Experiments
```bash
# Use full config for comprehensive evaluation
python cora_experiments.py --config config.yaml
```

### Parameter Exploration
1. Copy `config.yaml` to `config_custom.yaml`
2. Modify parameters of interest
3. Run: `python cora_experiments.py --config config_custom.yaml`

### Hyperparameter Search
1. Enable hyperparameter search in config
2. Define search space
3. Run experiments and analyze results

## Configuration Management in Python

You can also use the configuration system programmatically:

```python
from config_manager import ConfigManager, Config

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config("config.yaml")

# Access parameters
embedding_params = config_manager.get_embedding_params()
patch_params = config_manager.get_patch_params()
alignment_params = config_manager.get_alignment_params()

# Modify configuration
config.embedding.embedding_dim = 128
config.patches.num_patches = 20

# Save modified configuration
config_manager.save_config(config, "config_modified.yaml")
```

## Best Practices

### 1. Parameter Organization
- **Start with `config_simple.yaml`** for initial testing
- **Use `config.yaml`** for comprehensive experiments
- **Create custom configs** for specific research questions

### 2. Reproducibility
- Always set `random_seed` for reproducible results
- Use `num_runs > 1` to get statistical confidence
- Save configurations alongside results

### 3. Experiment Management
- Use descriptive `experiment.name` values
- Organize results by research question or parameter sweep
- Keep logs for debugging and analysis

### 4. Performance Optimization
- Use smaller configurations for development
- Start with fewer patches and epochs
- Scale up gradually for production runs

### 5. Result Analysis
- Check both individual run results and aggregated statistics
- Use visualizations to understand performance trends
- Compare across different embedding methods and configurations

## Troubleshooting

### Common Issues

1. **Configuration file not found**
   ```bash
   # Make sure you're in the experiments directory
   cd experiments
   python cora_experiments.py --config config.yaml
   ```

2. **Out of memory errors**
   ```yaml
   # Reduce parameters in config:
   embedding_dim: 32      # Smaller dimension
   hidden_dim: 16         # Smaller hidden layer
   num_patches: 5         # Fewer patches
   ```

3. **Slow experiments**
   ```yaml
   # Use config_simple.yaml or reduce:
   num_epochs: 50         # Fewer epochs
   num_runs: 1            # Single run for testing
   ```

4. **Invalid parameters**
   - Check the validation errors in the log
   - Refer to the valid options in the configuration comments
   - Use the ConfigManager validation to catch errors early

## Extending the System

### Adding New Embedding Methods
1. Add method name to `VALID_EMBEDDING_METHODS` in `config_manager.py`
2. Update `get_embedding_params()` method for method-specific parameters
3. Update configuration file with new method options

### Adding New Tasks
1. Create new configuration dataclass (e.g., `NodeClusteringConfig`)
2. Add task implementation in `cora_experiments.py`
3. Update aggregation and visualization functions

### Adding New Metrics
1. Add metric names to configuration evaluation_metrics lists
2. Implement metric computation in task functions
3. Update visualization functions if needed

This configuration system provides a robust foundation for systematic experimentation with the L2GX framework. It balances flexibility with ease of use, making it suitable for both rapid prototyping and comprehensive research studies.