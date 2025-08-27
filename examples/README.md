# L2GX Examples

This directory contains practical examples demonstrating how to use L2GX for graph embedding tasks.

## Embedding Examples

### Quick Demo

For a fast introduction to L2GX embedding methods:

```bash
python examples/simple_embedding_demo.py
```

This runs in ~10 seconds and demonstrates all three embedding approaches with reliable methods.

### Comprehensive Examples

For detailed examples with neural network methods:

```bash
python examples/embedding_examples.py
```

Note: This takes longer (~2-5 minutes) as it trains neural networks.

This script demonstrates:

1. **Simple Embedding**: Direct VGAE embedding of the full graph
2. **Patched L2G**: Graph patches with L2G alignment
3. **Patched Geo**: Graph patches with Geo alignment  
4. **Hierarchical**: Tree structure with hierarchical alignment

## What You'll See

The script will:
- Load the Cora citation network dataset
- Run each embedding method with optimized parameters
- Show timing and quality metrics for each approach
- Compare the results across methods

Example output:
```
L2GX Embedding Examples
=======================

Loading Cora dataset...
Loaded 2708 nodes, 10556 edges, 7 classes

============================================================
SIMPLE EMBEDDING EXAMPLE
============================================================
Using VGAE to embed the full graph...
✅ Simple embedding completed!
   Shape: (2708, 64)
   Mean norm: 2.145
   Time: 12.34 seconds

...

============================================================
EMBEDDING COMPARISON
============================================================
Method               Shape        Mean Norm    Std Norm    
------------------------------------------------------------
Simple VGAE          (2708, 64)   2.145        0.523       
Patched L2G          (2708, 64)   1.987        0.456       
Patched Geo          (2708, 64)   2.201        0.489       
Hierarchical         (2708, 64)   1.876        0.398       
```

## Understanding the Results

- **Shape**: All methods produce `(num_nodes, embedding_dim)` embeddings
- **Mean/Std Norm**: Statistics about embedding vector magnitudes
- **Time**: Computation time varies by method complexity
- **Patches**: Number and size of graph patches created

## Next Steps

After running the examples:

1. **Read the Documentation**: See `docs/source/embedding_guide.rst` for detailed explanations
2. **Try Configuration Files**: Use the scripts in `scripts/embedding/` for reproducible experiments
3. **Experiment with Parameters**: Modify embedding dimensions, patch sizes, alignment methods
4. **Apply to Your Data**: Adapt the examples to your own graph datasets

## Configuration-Based Experiments

For reproducible research, use the configuration-driven scripts:

```bash
# Patched embedding with L2G alignment
python scripts/embedding/patched_embedding_config.py config/patched_l2g_config.yaml

# Hierarchical embedding
python scripts/embedding/hierarchical_embedding_config.py config/hierarchical_l2g_config.yaml

# Multiple examples
python scripts/embedding/run_patched_examples.py
python scripts/embedding/run_hierarchical_examples.py
```

These scripts save detailed results, timing information, and experiment metadata for analysis.