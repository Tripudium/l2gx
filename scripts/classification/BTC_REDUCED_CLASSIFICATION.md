# BTC-Reduced Dataset Classification Documentation

This document provides comprehensive information about using the BTC-Reduced dataset for node classification experiments in the L2GX framework.

## Overview

The BTC-Reduced dataset contains Bitcoin transaction network data focused on entity type classification. It includes only nodes with known labels (entity types), making it a manageable subset of the full Bitcoin transaction graph while preserving the essential structure for financial entity classification tasks.

## Dataset Characteristics

### Basic Statistics
- **Dataset Name**: `btc-reduced` 
- **Total Nodes**: ~34,000 labeled entities (0.03% of full Bitcoin graph)
- **Edges**: ~10,000 transaction relationships between labeled entities
- **Node Features**: 8 features (transaction statistics, degrees, amounts)
- **Classes**: 11 Bitcoin entity types

### Entity Classes
The dataset classifies Bitcoin addresses into the following entity types:

1. **BET** (19.2%) - Gambling/betting services
2. **BRIDGE** (0.2%) - Cross-chain bridge services  
3. **EXCHANGE** (2.0%) - Cryptocurrency exchanges
4. **FAUCET** (0.3%) - Bitcoin faucet services
5. **INDIVIDUAL** (68.2%) - Individual users (majority class)
6. **MARKETPLACE** (0.4%) - Online marketplaces
7. **MINING** (2.7%) - Mining pools and services
8. **MIXER** (0.4%) - Transaction mixing services
9. **PONZI** (2.0%) - Ponzi scheme addresses
10. **RANSOMWARE** (0.6%) - Ransomware-related addresses
11. **OTHER** (4.1%) - Other/unknown entity types

### Graph Properties
- **Sparsity**: Much sparser than citation networks (Cora/PubMed)
- **Heterogeneity**: High diversity in node behavior patterns
- **Class Imbalance**: Severe imbalance with INDIVIDUAL as majority class
- **Financial Context**: Real-world financial transaction relationships

## Configuration Files

### 1. Full Classification Configuration
**File**: `btc_reduced_classification_config.yaml`

Complete experimental setup for comprehensive Bitcoin entity classification analysis.

**Key Features**:
- **5 Embedding Methods**: Full Graph VGAE, L2G Rademacher, Hierarchical L2G, L2G Standard, Geo Rademacher
- **Dimension Sweep**: [8, 16, 32, 64, 128, 256] for financial data complexity
- **Bitcoin-Optimized Parameters**:
  - 15 patches (between Cora:10 and PubMed:20)
  - Patched methods: 128/256 overlaps for transaction sparsity
  - Hierarchical: 256/512 overlaps for better sparse graph connectivity
  - Target patch degree 3 for sparse connectivity
  - More epochs (150) for financial relationship complexity
- **Imbalanced Class Handling**: Balanced class weights and weighted averaging
- **Financial Crime Focus**: Priority on EXCHANGE, MIXER, RANSOMWARE, PONZI classes
- **GraphSAGE Removed**: Performs poorly on extremely sparse BTC graph (0.04% edge density)

**Usage**:
```bash
python scripts/classification/classification_experiment.py scripts/classification/btc_reduced_classification_config.yaml
```

### 2. Simple Test Configuration
**File**: `btc_reduced_simple_config.yaml`

Simplified configuration for quick testing and development.

**Key Features**:
- **3 Methods**: Full Graph, L2G Rademacher, Hierarchical
- **3 Dimensions**: [32, 64, 128] for faster testing
- **Limited Dataset**: 5,000 nodes for quick iteration
- **Essential Setup**: Core Bitcoin classification functionality

**Usage**:
```bash
python scripts/classification/classification_experiment.py scripts/classification/btc_reduced_simple_config.yaml
```

### 3. Unified Framework Configuration
**File**: `btc_reduced_unified_config.yaml`

Modern unified approach using the latest L2GX embedding framework.

**Key Features**:
- **5 Methods**: Including GraphSAGE for heterogeneous Bitcoin data
- **Unified API**: Uses `get_embedding("patched", ...)` and `get_embedding("hierarchical", ...)`
- **Bitcoin-Specific Evaluation**: Financial crime detection metrics
- **Full Dataset**: All ~34K labeled nodes

**Usage**:
```bash
python scripts/classification/classification_experiment.py scripts/classification/btc_reduced_unified_config.yaml
```

## Bitcoin-Specific Optimizations

### Parameter Adjustments for Financial Data

#### Embedding Parameters
- **Epochs**: 150 (vs 100 for citation networks) - Complex financial relationships need more training
- **Patience**: 25 (vs 20) - Allow for longer convergence on sparse graphs
- **Learning Rate**: 0.001 - Conservative for stable financial pattern learning

#### Patching Strategy
- **Number of Patches**: 15 - Moderate for ~34K nodes (between Cora and PubMed)
- **Overlap Sizes**: 128/256 (vs 256/512) - Smaller for transaction graph sparsity
- **Target Degree**: 3 (vs 4-5) - Lower connectivity in Bitcoin transactions
- **Clustering**: METIS with resistance-based sparsification

#### Classification Setup
- **Multi-class Strategy**: One-vs-Rest for 11-class problem
- **Class Weighting**: Balanced to handle 68% INDIVIDUAL majority
- **Solver**: LBFGS with increased iterations (2000) for convergence
- **Evaluation**: Macro and weighted averaging for imbalanced classes

### Financial Crime Detection Focus

The configuration prioritizes detection of financially significant entity types:

- **High-Value Classes**: EXCHANGE, MIXER, RANSOMWARE, PONZI
- **Security Metrics**: Precision/recall for each crime-related class
- **Confusion Matrix**: Detailed per-class performance analysis
- **Priority Evaluation**: Separate metrics for financial crime categories

## Expected Performance

### Baseline Results
Initial testing with VGAE embeddings shows:
- **Overall Accuracy**: ~49% (reasonable for 11-class imbalanced problem)
- **Majority Class (INDIVIDUAL)**: High recall (75%), moderate precision
- **Minority Classes**: Variable performance due to class imbalance
- **Financial Crime Classes**: Decent precision for detection tasks

### Performance Considerations
- **Class Imbalance**: INDIVIDUAL dominates (68%), affecting overall metrics
- **Sparse Connectivity**: Lower graph connectivity than citation networks
- **Financial Complexity**: Real-world financial patterns are inherently noisy
- **Evaluation Focus**: Weighted F1-score more meaningful than raw accuracy

## Comparison with Other Datasets

### vs Cora Dataset
| Aspect | Cora | BTC-Reduced |
|--------|------|-------------|
| Nodes | 2,708 | ~34,000 |
| Classes | 7 | 11 |
| Domain | Academic citations | Financial transactions |
| Balance | Moderate imbalance | Severe imbalance (68% majority) |
| Connectivity | Dense | Sparse |
| Features | 1,433 (text) | 8 (financial) |

### vs PubMed Dataset  
| Aspect | PubMed | BTC-Reduced |
|--------|--------|-------------|
| Nodes | 19,717 | ~34,000 |
| Classes | 3 | 11 |
| Domain | Medical abstracts | Financial transactions |
| Balance | Moderate | Severe imbalance |
| Connectivity | Dense | Sparse |
| Features | 500 (text) | 8 (financial) |

## Usage Examples

### Quick Test Run
```bash
# Fast test with 3 methods and smaller dataset
python scripts/classification/classification_experiment.py scripts/classification/btc_reduced_simple_config.yaml
```

### Full Comparison Study
```bash
# Complete analysis with 6 methods across multiple dimensions
python scripts/classification/classification_experiment.py scripts/classification/btc_reduced_classification_config.yaml
```

### Modern Unified Framework
```bash
# Latest L2GX API with 5 methods including GraphSAGE
python scripts/classification/classification_experiment.py scripts/classification/btc_reduced_unified_config.yaml
```

### Custom Analysis
```python
# Direct API usage for custom experiments
from l2gx.datasets import get_dataset
from l2gx.embedding import get_embedding

# Load dataset
btc = get_dataset("btc-reduced", max_nodes=5000)
data = btc[0]

# Generate embedding
embedder = get_embedding("vgae", embedding_dim=64, epochs=100)
embedding = embedder.fit_transform(data)

# Use with standard sklearn classification pipeline
```

## Output and Results

### Generated Files
- **Raw Results**: CSV with all experimental data
- **Summary Results**: Aggregated performance metrics  
- **Accuracy Plots**: Visualization of method comparisons
- **Confusion Matrices**: Per-class performance analysis
- **Classification Reports**: Detailed precision/recall/F1 for each entity type
- **Experiment Report**: Comprehensive analysis document

### Key Metrics
- **Accuracy**: Overall classification accuracy
- **Weighted F1**: F1-score weighted by class frequency
- **Macro F1**: Unweighted average F1 across all classes
- **Per-Class Metrics**: Precision/recall for each Bitcoin entity type
- **Crime Detection**: Specific metrics for financial crime classes

## Implementation Notes

### Dataset Integration
The BTC-Reduced dataset is fully integrated into the L2GX framework:
- Uses standard `get_dataset("btc-reduced")` API
- Compatible with all embedding methods (VGAE, GraphSAGE, etc.)
- Works with patched and hierarchical approaches
- Supports TGraph conversion for all operations

### Technical Considerations
- **Memory Usage**: ~34K nodes manageable on standard hardware
- **Computation Time**: Moderate - between Cora and PubMed complexity
- **Class Imbalance**: Requires careful evaluation metric selection
- **Sparse Graphs**: Some embedding methods may need parameter adjustment

## Troubleshooting

### Common Issues
1. **Low Accuracy**: Expected due to class imbalance - focus on weighted metrics
2. **Memory Errors**: Reduce `max_nodes` parameter in dataset configuration
3. **Convergence Issues**: Increase epochs or adjust learning rate for sparse graphs
4. **Empty Patches**: May occur with small samples - reduce number of patches

### Optimization Tips
1. **Class Imbalance**: Always use `class_weight: "balanced"` in classification config
2. **Evaluation**: Focus on weighted F1 and per-class metrics, not raw accuracy
3. **Parameter Tuning**: Start with provided configurations and adjust gradually
4. **Financial Focus**: Prioritize precision for financial crime detection classes

## Future Enhancements

### Potential Improvements
- **Temporal Analysis**: Incorporate transaction timing information
- **Enhanced Features**: Additional financial metrics and graph statistics
- **Hierarchical Classification**: Multi-level entity type classification
- **Anomaly Detection**: Unsupervised detection of unusual transaction patterns
- **Cross-Dataset**: Comparison with other financial graph datasets

### Research Applications
- **Financial Crime Detection**: AML/KYC applications
- **Graph Neural Networks**: Benchmarking on real financial data
- **Imbalanced Learning**: Techniques for severely imbalanced graph classification
- **Sparse Graph Methods**: Algorithms optimized for low-connectivity graphs

---

*This documentation covers the complete BTC-Reduced dataset integration for node classification in L2GX. For questions or issues, refer to the main L2GX documentation or create GitHub issues.*