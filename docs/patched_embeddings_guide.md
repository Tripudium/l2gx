# Patched Embeddings Guide

This guide provides a comprehensive overview of how to compute patched embeddings from graphs using the L2GX framework. It covers the complete workflow from graph loading to final embedding, with configuration options explained at each step.

## Table of Contents

1. [Overview](#overview)
2. [Basic Workflow](#basic-workflow)
3. [Configuration Options](#configuration-options)
4. [Simple Examples](#simple-examples)
5. [Advanced Usage](#advanced-usage)
6. [Hierarchical Embeddings](#hierarchical-embeddings)
7. [Performance Tips](#performance-tips)
8. [Troubleshooting](#troubleshooting)

## Overview

The patched embedding approach divides large graphs into overlapping subgraphs (patches), computes local embeddings for each patch, and then aligns them to create a global embedding. This divide-and-conquer strategy enables:

- **Scalability**: Process graphs too large for single-machine memory
- **Parallelization**: Embed patches independently in parallel
- **Quality**: Preserve local structure while maintaining global coherence

### Core Workflow

```
Graph → Patches → Local Embeddings → Alignment → Global Embedding
```

## Basic Workflow

### Step 1: Load Your Graph

```python
from l2gx.datasets import get_dataset
from l2gx.graphs import TGraph

# Option A: Use built-in datasets
dataset = get_dataset("Cora")
data = dataset.to("torch-geometric")
graph = TGraph(data.edge_index, x=data.x)

# Option B: Load your own graph
import torch
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
graph = TGraph(edge_index, num_nodes=3)
```

### Step 2: Create Patches

```python
from l2gx.patch import create_patches

patch_graph = create_patches(
    graph=graph,
    num_patches=10,           # Number of patches to create
    clustering_method="metis", # Algorithm for partitioning
    min_overlap=64,           # Minimum nodes in overlap
    target_overlap=128        # Target overlap size
)

patches = patch_graph.patches
overlaps = patch_graph.overlap_nodes
```

### Step 3: Embed Patches

```python
from l2gx.embedding import get_embedding

embedder = get_embedding("vgae", embedding_dim=64, epochs=200)

for patch in patches:
    # Create subgraph for this patch
    patch_graph = graph.subgraph(torch.tensor(patch.nodes))
    patch_data = patch_graph.to_tg()
    
    # Compute embedding
    coordinates = embedder.fit_transform(patch_data)
    patch.coordinates = coordinates
```

### Step 4: Align Patches

```python
from l2gx.align import get_aligner

aligner = get_aligner("l2g")
aligner.align_patches(patch_graph)
global_embedding = aligner.get_aligned_embedding()
```

## Configuration Options

### Patch Generation Options

#### Clustering Methods
- **`metis`**: Balanced graph partitioning with minimal edge cuts (recommended)
- **`fennel`**: Streaming algorithm with load balancing
- **`louvain`**: Community-based clustering for natural groups
- **`spread`**: Degree-based spreading algorithm
- **`hierarchical`**: Multi-level clustering with size constraints

```python
# Example: Using different clustering methods
patch_graph_metis = create_patches(graph, clustering_method="metis")
patch_graph_louvain = create_patches(graph, clustering_method="louvain")
patch_graph_fennel = create_patches(graph, clustering_method="fennel")
```

#### Overlap Parameters
- **`min_overlap`**: Minimum number of shared nodes between connected patches
- **`target_overlap`**: Desired overlap size (algorithm tries to achieve this)
- **`target_patch_degree`**: Target number of neighboring patches

```python
# Smaller overlaps = faster, less accurate
sparse_patches = create_patches(
    graph,
    min_overlap=16,
    target_overlap=32,
    target_patch_degree=2
)

# Larger overlaps = slower, more accurate
dense_patches = create_patches(
    graph,
    min_overlap=128,
    target_overlap=256,
    target_patch_degree=6
)
```

#### Sparsification Methods
Reduce patch complexity while preserving structure:

```python
patch_graph = create_patches(
    graph,
    sparsify_method="resistance",  # Options: resistance, spanning_tree, edge_sampling
    sparsification_ratio=0.8,      # Keep 80% of edges
    use_conductance_weighting=True # Weight by conductance
)
```

### Embedding Methods

All methods follow the same interface:

```python
# SVD - Classical spectral approach
svd_embedder = get_embedding("svd", embedding_dim=32)

# GAE - Graph Auto-Encoder
gae_embedder = get_embedding("gae", embedding_dim=64, epochs=500)

# VGAE - Variational Graph Auto-Encoder (recommended)
vgae_embedder = get_embedding(
    "vgae",
    embedding_dim=128,
    hidden_dim=256,
    epochs=200,
    learning_rate=0.001,
    patience=20  # Early stopping
)

# GraphSAGE - Inductive embedding
sage_embedder = get_embedding(
    "graphsage",
    embedding_dim=64,
    aggregator="mean",  # Options: mean, gcn, lstm
    num_layers=2
)

# DGI - Deep Graph Infomax
dgi_embedder = get_embedding(
    "dgi",
    embedding_dim=128,
    corruption_method="shuffle"
)
```

### Alignment Methods

#### L2G (Local2Global)
Original eigenvalue synchronization approach:

```python
# Standard L2G
l2g_standard = get_aligner("l2g", randomized_method="standard")

# Randomized L2G (faster for large dimensions)
l2g_random = get_aligner(
    "l2g",
    randomized_method="randomized",
    sketch_method="rademacher"  # Options: gaussian, rademacher, fourier
)

# Sparse-aware L2G
l2g_sparse = get_aligner("l2g", randomized_method="sparse_aware")
```

#### Geometric Alignment
Optimization on manifolds:

```python
geo_aligner = get_aligner(
    "geo",
    max_iter=100,
    tol=1e-6,
    use_manifold_optimization=True
)
```

## Simple Examples

### Example 1: Basic Patched Embedding

```python
from l2gx.datasets import get_dataset
from l2gx.graphs import TGraph
from l2gx.patch import create_patches
from l2gx.embedding import get_embedding
from l2gx.align import get_aligner
import torch

# Load dataset
dataset = get_dataset("Cora")
data = dataset.to("torch-geometric")
graph = TGraph(data.edge_index, x=data.x)

# Create patches
patch_graph = create_patches(graph, num_patches=10)

# Embed patches
embedder = get_embedding("vgae", embedding_dim=64)
for patch in patch_graph.patches:
    subgraph = graph.subgraph(torch.tensor(patch.nodes))
    patch.coordinates = embedder.fit_transform(subgraph.to_tg())

# Align
aligner = get_aligner("l2g")
aligner.align_patches(patch_graph)
embedding = aligner.get_aligned_embedding()

print(f"Final embedding shape: {embedding.shape}")
```

### Example 2: Customized Pipeline

```python
# Custom configuration for large graphs
patch_graph = create_patches(
    graph,
    num_patches=20,
    clustering_method="fennel",  # Fast streaming algorithm
    min_overlap=32,              # Smaller overlaps for speed
    target_overlap=64,
    sparsify_method="resistance",
    sparsification_ratio=0.7
)

# Fast embedding with early stopping
embedder = get_embedding(
    "gae",  # Faster than VGAE
    embedding_dim=32,
    epochs=100,
    patience=10
)

for i, patch in enumerate(patch_graph.patches):
    print(f"Processing patch {i+1}/{len(patch_graph.patches)}")
    subgraph = graph.subgraph(torch.tensor(patch.nodes))
    patch.coordinates = embedder.fit_transform(subgraph.to_tg())

# Randomized alignment for speed
aligner = get_aligner(
    "l2g",
    randomized_method="randomized",
    sketch_method="rademacher"
)
aligner.align_patches(patch_graph)
embedding = aligner.get_aligned_embedding()
```

### Example 3: Quality-Focused Configuration

```python
# Configuration prioritizing quality over speed
patch_graph = create_patches(
    graph,
    num_patches=15,
    clustering_method="metis",     # Optimal partitioning
    min_overlap=128,               # Large overlaps
    target_overlap=256,
    target_patch_degree=6,         # More connections
    use_conductance_weighting=True
)

# High-quality embeddings
embedder = get_embedding(
    "vgae",
    embedding_dim=256,
    hidden_dim=512,
    epochs=1000,
    learning_rate=0.0005,
    patience=50
)

for patch in patch_graph.patches:
    subgraph = graph.subgraph(torch.tensor(patch.nodes))
    patch.coordinates = embedder.fit_transform(subgraph.to_tg())

# Standard alignment (most accurate)
aligner = get_aligner("l2g", randomized_method="standard")
aligner.align_patches(patch_graph)
embedding = aligner.get_aligned_embedding()
```

## Advanced Usage

### Parallel Patch Embedding

```python
from concurrent.futures import ProcessPoolExecutor
import torch

def embed_patch(patch_nodes, graph, embedder_config):
    """Embed a single patch"""
    from l2gx.embedding import get_embedding
    
    embedder = get_embedding(**embedder_config)
    subgraph = graph.subgraph(torch.tensor(patch_nodes))
    return embedder.fit_transform(subgraph.to_tg())

# Parallel processing
embedder_config = {"method": "vgae", "embedding_dim": 64}
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = []
    for patch in patch_graph.patches:
        future = executor.submit(embed_patch, patch.nodes, graph, embedder_config)
        futures.append(future)
    
    for patch, future in zip(patch_graph.patches, futures):
        patch.coordinates = future.result()
```

### Custom Patch Post-Processing

```python
# Add noise for robustness testing
import numpy as np
rng = np.random.default_rng(42)

for patch in patch_graph.patches:
    noise = rng.normal(0, 0.01, patch.coordinates.shape)
    patch.coordinates += noise

# Scale patches
for patch in patch_graph.patches:
    patch.coordinates *= 2.0
```

### Evaluating Alignment Quality

```python
from l2gx.align import procrustes_error, local_error

# If you have ground truth positions
ground_truth = ...  # Your reference embedding
error = procrustes_error(embedding, ground_truth)
print(f"Procrustes error: {error:.4f}")

# Local consistency error
local_err = local_error(aligner.patches)
print(f"Local error: {local_err:.4f}")
```

## Hierarchical Embeddings

For very large graphs, use hierarchical decomposition:

```python
from l2gx.embedding.hierarchical_embedding import HierarchicalEmbedding

hierarchical_embedder = HierarchicalEmbedding(
    k_patches=4,              # Patches per level
    max_patch_size=200,       # When to subdivide
    embedding_dim=128,
    embedding_method="vgae",
    alignment_method="l2g",
    clustering_method="metis",
    min_overlap=64,
    target_overlap=128,
    verbose=True
)

# Automatically handles the full pipeline
embedding = hierarchical_embedder.fit(graph)

# Access statistics
stats = hierarchical_embedder.embedding_stats
print(f"Tree depth: {stats['max_depth']}")
print(f"Total patches: {stats['total_patches']}")
print(f"Leaf patches: {stats['leaf_patches']}")
```

### Hierarchical Configuration Examples

```python
# Shallow tree for medium graphs
shallow_config = HierarchicalEmbedding(
    k_patches=10,           # More patches per level
    max_patch_size=500,     # Larger patches before split
    embedding_dim=64
)

# Deep tree for huge graphs
deep_config = HierarchicalEmbedding(
    k_patches=2,            # Binary tree
    max_patch_size=100,     # Small patches
    embedding_dim=32,       # Lower dimension for speed
    embedding_method="gae"  # Faster method
)

# Balanced configuration
balanced_config = HierarchicalEmbedding(
    k_patches=4,
    max_patch_size=200,
    embedding_dim=128,
    embedding_method="vgae",
    min_overlap=32,         # Smaller overlaps for hierarchical
    target_overlap=64
)
```

## Performance Tips

### Memory Optimization

1. **Use sparse matrices**: The framework automatically handles sparse representations
2. **Reduce embedding dimension**: Start with 32-64 dimensions
3. **Limit patch size**: Keep patches under 1000 nodes
4. **Use sparsification**: Remove redundant edges

```python
# Memory-efficient configuration
patch_graph = create_patches(
    graph,
    num_patches=20,
    sparsify_method="resistance",
    sparsification_ratio=0.5  # Keep only 50% of edges
)

embedder = get_embedding("gae", embedding_dim=32)  # Small dimension
```

### Speed Optimization

1. **Use randomized methods**: Much faster for high dimensions
2. **Parallel processing**: Embed patches in parallel
3. **Early stopping**: Use patience parameter
4. **Rust implementations**: Use `fennel_rust` when available

```python
# Speed-optimized configuration
patch_graph = create_patches(
    graph,
    clustering_method="fennel_rust" if is_rust_available() else "fennel",
    min_overlap=16  # Minimal overlap
)

# Fast embedding
embedder = get_embedding(
    "gae",
    epochs=50,
    patience=5
)

# Fast alignment
aligner = get_aligner(
    "l2g",
    randomized_method="randomized",
    sketch_method="rademacher"
)
```

### Quality Optimization

1. **Increase overlaps**: Better alignment accuracy
2. **Use VGAE**: Better than GAE for most tasks
3. **More epochs**: Improve embedding quality
4. **METIS clustering**: Optimal graph partitioning

```python
# Quality-optimized configuration
patch_graph = create_patches(
    graph,
    clustering_method="metis",
    min_overlap=256,
    target_overlap=512,
    target_patch_degree=8
)

embedder = get_embedding(
    "vgae",
    embedding_dim=256,
    epochs=1000,
    learning_rate=0.0001
)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "No overlap between patches"
**Problem**: Patches don't share enough nodes for alignment.
**Solution**: Increase `min_overlap` and `target_overlap`:
```python
patch_graph = create_patches(graph, min_overlap=64, target_overlap=128)
```

#### 2. "Memory overflow during embedding"
**Problem**: Patches are too large.
**Solution**: Use more patches or hierarchical approach:
```python
# More, smaller patches
patch_graph = create_patches(graph, num_patches=30)

# Or use hierarchical
hierarchical = HierarchicalEmbedding(max_patch_size=100)
```

#### 3. "Poor alignment quality"
**Problem**: Embedding doesn't preserve structure.
**Solutions**:
- Increase overlap sizes
- Use more training epochs
- Try different embedding methods
- Check patch connectivity

```python
# Diagnose connectivity
import networkx as nx
patch_nx = patch_graph.to_networkx()
print(f"Connected: {nx.is_connected(patch_nx)}")
print(f"Components: {nx.number_connected_components(patch_nx)}")
```

#### 4. "Slow performance"
**Problem**: Pipeline takes too long.
**Solutions**:
- Use randomized methods
- Reduce embedding dimensions
- Enable parallel processing
- Use Rust implementations

```python
# Fast configuration
from l2gx.patch.clustering import is_rust_available

clustering = "fennel_rust" if is_rust_available() else "fennel"
patch_graph = create_patches(graph, clustering_method=clustering)

aligner = get_aligner("l2g", randomized_method="randomized")
```

### Validation Checks

```python
def validate_patched_embedding(patch_graph, embedding):
    """Validate the patched embedding pipeline"""
    
    # Check patch coverage
    all_nodes = set()
    for patch in patch_graph.patches:
        all_nodes.update(patch.nodes)
    print(f"Node coverage: {len(all_nodes)}/{patch_graph.num_nodes}")
    
    # Check overlaps
    overlaps = patch_graph.overlap_nodes
    overlap_sizes = [len(nodes) for nodes in overlaps.values()]
    if overlap_sizes:
        print(f"Overlap sizes: min={min(overlap_sizes)}, "
              f"max={max(overlap_sizes)}, mean={np.mean(overlap_sizes):.1f}")
    
    # Check embedding
    print(f"Embedding shape: {embedding.shape}")
    print(f"Contains NaN: {np.isnan(embedding).any()}")
    print(f"Contains Inf: {np.isinf(embedding).any()}")
    
    return True

# Use it
validate_patched_embedding(patch_graph, embedding)
```

## Summary

The patched embedding approach offers flexible configurations for different use cases:

- **For small graphs (<10K nodes)**: Use standard methods with large overlaps
- **For medium graphs (10K-1M nodes)**: Use patched approach with 10-50 patches
- **For large graphs (>1M nodes)**: Use hierarchical approach or many patches with randomized methods

Key decisions:
1. **Number of patches**: More patches = better scalability but harder alignment
2. **Overlap size**: Larger overlaps = better quality but higher memory/time
3. **Embedding method**: VGAE for quality, GAE/SVD for speed
4. **Alignment method**: Standard for accuracy, randomized for speed

Start with the basic configuration and adjust based on your specific requirements for speed, memory, and quality.