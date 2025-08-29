# Bitcoin Entity Classification Scripts

This directory contains all scripts and configurations for working with Bitcoin transaction graph datasets in the L2GX framework, specifically focused on entity type classification.

## 📁 Directory Contents

### Core Dataset Implementation
- **`btc.py`** - Bitcoin dataset implementation (BTCDataset and BTCReducedDataset classes)

### Classification Experiments
- **`classification_experiment.py`** - Standard classification experiment runner
- **`classification_experiment_balanced.py`** - Enhanced runner with class imbalance handling

### Configuration Files
- **`btc_reduced_classification_config.yaml`** - Full 5-method comparison config
- **`btc_reduced_simple_config.yaml`** - Quick 3-method test config  
- **`btc_reduced_unified_config.yaml`** - Modern unified framework config
- **`btc_reduced_balanced_config.yaml`** - Class imbalance handling config

### Visualization and Analysis
- **`create_btc_plots.py`** - Generate accuracy vs dimension plots
- **`visualize_embeddings.py`** - UMAP visualization of embeddings
- **`diagnose_embeddings.py`** - Check for degenerate/collapsed embeddings

### Testing and Validation
- **`test_btc_classification.py`** - Test classification pipeline
- **`test_btc_tgraph_fixed.py`** - Test TGraph conversion compatibility
- **`test_sampling_strategies.py`** - Quick test of sampling methods

### Documentation
- **`BTC_REDUCED_CLASSIFICATION.md`** - Comprehensive documentation
- **`README.md`** - This file

## 🚀 Quick Start

### 1. Simple Test
```bash
python test_btc_classification.py
```

### 2. Quick Classification Experiment
```bash
python classification_experiment.py btc_reduced_simple_config.yaml
```

### 3. Full Comparison Study
```bash
python classification_experiment.py btc_reduced_classification_config.yaml
```

### 4. Class Imbalance Handling
```bash
python classification_experiment_balanced.py btc_reduced_balanced_config.yaml
```

### 5. Generate Plots
```bash
python create_btc_plots.py <results_directory>
```

## 📊 Dataset Information

### BTC-Reduced Dataset
- **Size**: ~34,000 labeled Bitcoin addresses (0.03% of full graph)
- **Classes**: 11 entity types
- **Features**: 8 transaction-based features
- **Imbalance**: Severe (68% INDIVIDUAL, minority classes <1%)

### Entity Classes
1. **INDIVIDUAL** (68.2%) - Individual users
2. **BET** (19.2%) - Gambling services
3. **MINING** (2.7%) - Mining pools
4. **EXCHANGE** (2.0%) - Cryptocurrency exchanges
5. **PONZI** (2.0%) - Ponzi schemes
6. **OTHER** (4.1%) - Other entity types
7. **MIXER** (0.4%) - Transaction mixers
8. **MARKETPLACE** (0.4%) - Online marketplaces
9. **FAUCET** (0.3%) - Bitcoin faucets
10. **BRIDGE** (0.2%) - Cross-chain bridges
11. **RANSOMWARE** (0.6%) - Ransomware addresses

## 🔧 Configuration Options

### Standard Methods
- **Full Graph VGAE** - Baseline whole-graph embedding
- **L2G Rademacher** - Patched embedding with L2G alignment
- **L2G Standard** - Patched embedding without randomization
- **Hierarchical L2G** - Binary tree hierarchical approach
- **Geo Rademacher** - Patched embedding with geometric alignment

### Bitcoin-Specific Optimizations
- **Increased epochs**: 150 (vs 100) for complex financial relationships
- **Sparse-aware patching**: Lower overlap sizes and degrees
- **Hierarchical tuning**: Increased overlaps (256/512) for sparse graphs
- **GraphSAGE removed**: Poor performance on 0.04% edge density

### Class Imbalance Solutions
- **Sampling strategies**: SMOTE, ROS, RUS, ADASYN, combined methods
- **Balanced metrics**: Focus on balanced accuracy and F1 macro
- **Class weighting**: Built-in balanced class weights
- **Minority class focus**: Priority tracking for crime-related entities

## 📈 Expected Performance

### Typical Results
- **Full Graph VGAE**: ~48-52% accuracy, good baseline
- **L2G Methods**: ~45-55% accuracy, consistent across dimensions
- **Hierarchical**: Variable (17-77%), improved with larger overlaps
- **With sampling**: +10-30% improvement in balanced accuracy

### Key Metrics
- **Accuracy**: Standard metric (biased by majority class)
- **Balanced Accuracy**: Average recall across all classes
- **F1 Macro**: Unweighted mean F1 (treats all classes equally)
- **Per-class metrics**: Essential for minority crime detection

## 🏗️ Architecture

### Dataset Pipeline
1. Load labeled nodes from parquet files
2. Filter to labeled nodes only (~34K from 252M total)
3. Create consecutive node ID mapping
4. Build sparse transaction edge graph
5. Apply log transforms to financial features

### Embedding Pipeline
1. Convert to TGraph format
2. Apply chosen embedding method
3. Generate node embeddings
4. Apply sampling strategy (if configured)
5. Train classifier with balanced metrics

### Analysis Pipeline
1. Track multiple metrics (accuracy, balanced accuracy, F1)
2. Generate dimension vs accuracy plots
3. Save detailed per-class performance
4. Create summary reports

## ⚠️ Common Issues

### Performance Issues
- **Low accuracy**: Expected due to severe class imbalance and sparse graph
- **Constant accuracy**: GraphSAGE fails on sparse graphs (removed)
- **Unstable hierarchical**: Fixed with increased overlaps

### Memory Issues
- **Large datasets**: Use `max_nodes` parameter to limit size
- **SMOTE failure**: Automatically adjusts k-neighbors for small classes

### Graph Issues
- **Sparse connectivity**: Only 0.04% edge density between labeled nodes
- **Disconnected components**: Some sampling may create isolated nodes

## 🔍 Troubleshooting

### Debugging Steps
1. Run `test_btc_classification.py` to verify basic functionality
2. Use `diagnose_embeddings.py` to check for collapsed embeddings
3. Try `test_sampling_strategies.py` to verify imbalance handling
4. Check `visualize_embeddings.py` for embedding quality

### Configuration Issues
- **Path errors**: All scripts assume they're run from project root
- **Missing dependencies**: Install imbalanced-learn for sampling strategies
- **CUDA errors**: Set device appropriately or use CPU-only mode

## 📚 References

- **Dataset**: Bitcoin transaction graph with entity labels
- **Methods**: L2GX embedding framework with graph alignment
- **Imbalance**: Imbalanced-learn library for sampling strategies
- **Evaluation**: Scikit-learn metrics with focus on balanced accuracy

For detailed technical information, see `BTC_REDUCED_CLASSIFICATION.md`.