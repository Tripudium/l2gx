# Node Classification

This directory contains a complete pipeline for node classification using graph embeddings generated from the embedding pipeline.

## Overview

The classification pipeline:
1. Loads or generates embeddings using embedding configuration files
2. Splits data into train/validation/test sets
3. Trains various classifiers (Logistic Regression, Random Forest, SVM)
4. Evaluates performance with multiple metrics
5. Optionally performs cross-validation

## Files

### Scripts
- `run_classification.py` - Main classification pipeline script

### Configuration Templates
- `classification_config.yaml` - Default logistic regression configuration
- `classification_config_rf.yaml` - Random forest configuration
- `classification_config_svm.yaml` - SVM configuration

## Usage

### Basic Usage (with default logistic regression)

```bash
# Use L2G embeddings
python run_classification.py ../embedding/embedding_config_l2g.yaml

# Use Geo embeddings
python run_classification.py ../embedding/embedding_config_geo.yaml
```

### With Custom Classifier Configuration

```bash
# Use Random Forest classifier
python run_classification.py ../embedding/embedding_config_l2g.yaml --classifier classification_config_rf.yaml

# Use SVM classifier
python run_classification.py ../embedding/embedding_config_geo.yaml --classifier classification_config_svm.yaml
```

### Using Pre-computed Embeddings

If embeddings have already been computed (from previous runs), the pipeline will automatically load them from the output directory specified in the embedding config.

## Configuration Options

### Classifier Types

1. **Logistic Regression** (default)
   - Fast and interpretable
   - Good baseline for linear relationships
   - Works well with high-dimensional data

2. **Random Forest**
   - Handles non-linear relationships
   - Provides feature importance
   - No feature scaling needed

3. **SVM (Support Vector Machine)**
   - Good for complex decision boundaries
   - Works well with high-dimensional data
   - Requires feature scaling

### Evaluation Options

- **Train/Val/Test Split**: Configurable ratios (default: 70/10/20)
- **Stratified Splitting**: Maintains class distribution
- **Cross-Validation**: Optional k-fold CV for robust evaluation

### Preprocessing Options

- **Feature Scaling**: StandardScaler normalization (important for SVM and LR)
- **Class Balancing**: Optional (not yet implemented)

## Output

Results are saved in `results/classification_YYYYMMDD_HHMMSS/` with:
- `results.yaml` - Complete results including all metrics
- `summary.txt` - Human-readable summary of key results

### Metrics Reported

- **Accuracy**: Overall correct predictions
- **F1 Score**: 
  - Macro: Unweighted mean across classes
  - Micro: Global calculation
  - Weighted: Weighted by class support
- **Per-class metrics**: Precision, recall, F1
- **Confusion matrix**
- **Cross-validation scores** (if enabled)

## Examples

### Example 1: Quick Evaluation with L2G

```bash
python run_classification.py ../embedding/embedding_config_l2g.yaml
```

Expected output:
```
NODE CLASSIFICATION PIPELINE
============================================================
Loading existing embeddings from results/l2g_experiment/embedding_results.npz
Embeddings shape: (2708, 128)
Number of classes: 7

Test Results:
  Accuracy: 0.8463
  F1 (macro): 0.8234
  F1 (weighted): 0.8445
```

### Example 2: Comparing Classifiers

```bash
# Logistic Regression
python run_classification.py ../embedding/embedding_config_l2g.yaml

# Random Forest
python run_classification.py ../embedding/embedding_config_l2g.yaml --classifier classification_config_rf.yaml

# SVM
python run_classification.py ../embedding/embedding_config_l2g.yaml --classifier classification_config_svm.yaml
```

### Example 3: Full Pipeline with New Embeddings

```bash
# Remove old embeddings to force regeneration
rm -rf results/l2g_experiment/

# Run classification (will generate embeddings first)
python run_classification.py ../embedding/embedding_config_l2g.yaml
```

## Tips

1. **For quick experiments**: Use logistic regression (default)
2. **For best accuracy**: Try SVM with RBF kernel
3. **For interpretability**: Use logistic regression or random forest
4. **For large datasets**: Use logistic regression with 'saga' solver
5. **Always scale features** when using SVM or logistic regression

## Extending the Pipeline

To add new classifiers:
1. Add the classifier type in `create_classifier()` method
2. Create a new configuration template
3. Add any classifier-specific preprocessing if needed

To add new metrics:
1. Modify the `evaluate_model()` method
2. Add the metric calculation
3. Update the summary output format