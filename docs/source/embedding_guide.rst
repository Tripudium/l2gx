Graph Embedding Guide
=====================

L2GX provides powerful graph embedding capabilities through three main approaches:

1. **Simple Embedding**: Direct embedding of the full graph using standard methods
2. **Patched Embedding**: Decompose graph into patches, embed locally, and align globally  
3. **Hierarchical Embedding**: Create tree structure of patches and align bottom-up

The **patched** and **hierarchical** embedding methods require an **aligner object** to be explicitly provided, giving you full control over the alignment process.

Quick Start
-----------

All examples start with loading a dataset:

.. code-block:: python

   from l2gx.datasets import get_dataset
   from l2gx.embedding import get_embedding
   from l2gx.align import get_aligner
   from l2gx.graphs import TGraph
   
   # Load a dataset
   dataset = get_dataset("Cora")
   pg_data = dataset.to("torch-geometric")
   data = TGraph.from_tg(pg_data)
   print(f"Loaded {data.num_nodes} nodes, {data.num_edges} edges")

Patched Embedding
-----------------

The **patched embedding** approach decomposes graphs into overlapping patches, embeds each patch locally, then aligns them globally using sophisticated alignment methods. This is ideal for large graphs and provides excellent scalability.

.. code-block:: python

   # Create and configure L2G aligner
   l2g_aligner = get_aligner("l2g")
   l2g_aligner.randomized_method = "randomized"
   l2g_aligner.sketch_method = "rademacher"
   
   # Create patched embedder using get_embedding
   embedder = get_embedding(
       "patched",                  # Use the patched embedding method
       embedding_dim=128,
       aligner=l2g_aligner,        # Required: pre-configured aligner
       num_patches=10,             # Number of patches to create
       base_method="vgae",         # Method for embedding each patch
       clustering_method="metis",  # Graph partitioning method
       epochs=200,                 # Parameters for base method
       learning_rate=0.001
   )
   
   # Compute embedding
   embedding = embedder.fit_transform(data.to_tg())
   print(f"Embedding shape: {embedding.shape}")

Hierarchical Embedding
-----------------------

The **hierarchical embedding** approach creates a tree structure of graph partitions, embeds leaf nodes, then aligns embeddings bottom-up through the tree. 

**Important Note**: The hierarchical embedding intelligently selects alignment methods:

- **Binary trees** (``branching_factor=2``): Uses Procrustes alignment by default for stability
- **Multi-way trees** (``branching_factor>2``): Uses the provided aligner configuration (L2G, Geo) with automatic fallback to Procrustes if needed

The system extracts parameters from your provided aligner and creates fresh instances for each alignment operation to avoid state conflicts.

.. code-block:: python

   # Create L2G aligner (will be used for ternary tree)
   l2g_aligner = get_aligner("l2g")
   l2g_aligner.randomized_method = "randomized"
   
   # Create hierarchical embedder using get_embedding
   embedder = get_embedding(
       "hierarchical",             # Use the hierarchical embedding method
       embedding_dim=128,
       aligner=l2g_aligner,        # Required: L2G aligner configuration
       max_patch_size=500,         # Maximum nodes in leaf patches
       branching_factor=3,         # Children per internal node (ternary tree, uses L2G)
       base_method="vgae",         # Method for embedding leaf patches
       clustering_method="metis",  # Graph partitioning method
       epochs=200,                 # Parameters for base method
       learning_rate=0.001
   )
   
   # Compute embedding (L2G alignment will be used for ternary splits)
   embedding = embedder.fit_transform(data.to_tg())
   print(f"Embedding shape: {embedding.shape}")
   
   # Get tree structure information
   tree_info = embedder.get_tree_structure()
   print(f"Tree depth: {tree_info['max_depth']}")
   print(f"Number of leaves: {tree_info['num_leaves']}")

Simple Embedding
----------------

For direct embedding of the full graph using standard methods (no aligner required):

.. code-block:: python

   # Create embedder for full graph
   embedder = get_embedding(
       "vgae",                    # Method: vgae, gae, svd, dgi, graphsage
       embedding_dim=128,
       epochs=200,
       learning_rate=0.001,
       patience=15
   )
   
   # Compute embedding
   embedding = embedder.fit_transform(data.to_tg())
   print(f"Embedding shape: {embedding.shape}")  # (num_nodes, embedding_dim)

Available simple embedding methods:

- **vgae**: Variational Graph Auto-Encoder (neural, requires training)
- **gae**: Graph Auto-Encoder (neural, requires training) 
- **svd**: Singular Value Decomposition (fast, no training)
- **dgi**: Deep Graph Infomax (self-supervised, requires training)
- **graphsage**: GraphSAGE (inductive, requires training)

Patched Embedding
-----------------

Decomposes graphs into overlapping patches, embeds each patch locally, then aligns them globally using sophisticated alignment methods.

L2G Alignment Example
~~~~~~~~~~~~~~~~~~~~~

L2G (Local-to-Global) alignment uses randomized sketching techniques:

.. code-block:: python

   # Create and configure L2G aligner
   l2g_aligner = get_aligner("l2g")
   l2g_aligner.randomized_method = "randomized"  # Enable randomization
   l2g_aligner.sketch_method = "rademacher"      # Sketch method
   
   # Create patched embedder
   embedder = get_embedding(
       "patched",
       embedding_dim=128,
       aligner=l2g_aligner,        # Required aligner object
       num_patches=10,             # Number of patches to create
       base_method="vgae",         # Method for embedding each patch
       clustering_method="metis",  # Graph partitioning method
       min_overlap=10,             # Minimum overlap between patches
       target_overlap=20,          # Target overlap size
       epochs=200,                 # Base method parameters
       learning_rate=0.001,
       patience=15
   )
   
   # Compute embedding
   embedding = embedder.fit_transform(data.to_tg())
   print(f"Embedding shape: {embedding.shape}")

Geo Alignment Example
~~~~~~~~~~~~~~~~~~~~~

Geometric alignment uses manifold optimization:

.. code-block:: python

   # Create and configure Geo aligner
   geo_aligner = get_aligner("geo", 
       method="orthogonal",        # orthogonal or euclidean
       use_scale=True,             # Enable scale optimization
       num_epochs=10,              # Optimization epochs
       learning_rate=0.01          # Learning rate
   )
   
   # Create patched embedder with Geo alignment
   embedder = get_embedding(
       "patched",
       embedding_dim=128,
       aligner=geo_aligner,        # Custom configured aligner
       num_patches=8,              # Fewer patches work better with Geo
       base_method="svd",          # Fast base method
       min_overlap=15,             # Higher overlap for better alignment
       target_overlap=30
   )
   
   embedding = embedder.fit_transform(data.to_tg())

Hierarchical Embedding
-----------------------

Creates a tree structure of graph partitions, embeds leaf nodes, and aligns embeddings bottom-up through the tree structure.

Basic Hierarchical Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create aligner (currently hierarchical uses Procrustes regardless)
   l2g_aligner = get_aligner("l2g")
   
   # Create hierarchical embedder
   embedder = get_embedding(
       "hierarchical",
       embedding_dim=128,
       aligner=l2g_aligner,        # Required (but Procrustes is used)
       max_patch_size=500,         # Maximum nodes in leaf patches
       branching_factor=3,         # Children per internal node (ternary tree)
       base_method="vgae",         # Method for embedding leaf patches
       clustering_method="metis",  # Graph partitioning method
       max_levels=5,               # Maximum tree depth
       epochs=200,                 # Base method parameters
       learning_rate=0.001,
       patience=15
   )
   
   embedding = embedder.fit_transform(data.to_tg())
   print(f"Embedding shape: {embedding.shape}")

Advanced Hierarchical Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # More detailed hierarchical configuration with Geo aligner
   geo_aligner = get_aligner("geo", method="euclidean", use_scale=False)
   
   embedder = get_embedding(
       "hierarchical", 
       embedding_dim=96,
       aligner=geo_aligner,        # Geo aligner parameters will be extracted
       max_patch_size=300,         # Smaller leaf patches
       min_patch_size=75,          # Minimum patch size (default: max_patch_size // 4)
       branching_factor=2,         # Binary tree (uses Procrustes by default)
       base_method="svd",          # Fast method for leaves
       clustering_method="metis",
       max_levels=4,               # Limit tree depth
       min_overlap=64,             # Overlap between patches
       target_overlap=128,
       verbose=True                # Show tree construction
   )
   
   embedding = embedder.fit_transform(data.to_tg())
   
   # Get tree structure information
   tree_info = embedder.get_tree_structure()
   print(f"Tree depth: {tree_info['max_depth']}")
   print(f"Number of leaves: {tree_info['num_leaves']}")

Forcing Multi-way Alignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To force usage of L2G or Geo alignment instead of the Procrustes default, use ``branching_factor > 2``:

.. code-block:: python

   # Force L2G alignment by using ternary tree
   l2g_aligner = get_aligner("l2g", randomized_method="randomized")
   
   embedder = get_embedding(
       "hierarchical",
       embedding_dim=64,
       aligner=l2g_aligner,
       branching_factor=3,         # Forces L2G alignment usage
       max_patch_size=400,
       base_method="svd",
       verbose=True
   )
   
   # Force Geo alignment by using quaternary tree  
   geo_aligner = get_aligner("geo", method="orthogonal", use_scale=True)
   
   embedder = get_embedding(
       "hierarchical",
       embedding_dim=64, 
       aligner=geo_aligner,
       branching_factor=4,         # Forces Geo alignment usage
       max_patch_size=300,
       base_method="svd"
   )

Configuration-Based Experiments
--------------------------------

L2GX provides configuration-driven experiment scripts for reproducible research:

Patched Embedding with Config Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a YAML configuration file:

.. code-block:: yaml

   # patched_l2g_config.yaml
   experiment:
     name: "Patched_L2G_Experiment"
     output_dir: "results/patched_l2g"
   
   dataset:
     name: "Cora"
     normalize_features: false
   
   patched:
     num_patches: 10
     base_method: "vgae"
     embedding_dim: 128
     epochs: 200
     learning_rate: 0.001
   
   aligner:
     method: "l2g"
     randomized_method: "randomized"
     sketch_method: "rademacher"

Run the experiment:

.. code-block:: bash

   python scripts/embedding/patched_embedding_config.py config/patched_l2g_config.yaml

Hierarchical Embedding with Config Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # hierarchical_l2g_config.yaml
   experiment:
     name: "Hierarchical_L2G_Experiment"
     output_dir: "results/hierarchical_l2g"
   
   dataset:
     name: "Cora"
   
   hierarchical:
     max_patch_size: 500
     branching_factor: 3
     base_method: "vgae"
     embedding_dim: 128
     epochs: 200
   
   aligner:
     method: "l2g"
     randomized_method: "randomized"

.. code-block:: bash

   python scripts/embedding/hierarchical_embedding_config.py config/hierarchical_l2g_config.yaml

Available Aligners
------------------

L2G (Local-to-Global)
~~~~~~~~~~~~~~~~~~~~~

Best for: Large numbers of patches, scalable alignment

.. code-block:: python

   aligner = get_aligner("l2g")
   aligner.randomized_method = "randomized"    # or "deterministic"
   aligner.sketch_method = "rademacher"        # or "gaussian"
   aligner.sketch_dimension = 100              # Optional: sketch size
   aligner.regularization = 1e-6               # Optional: regularization
   aligner.max_iterations = 1000               # Optional: max iterations
   aligner.tolerance = 1e-8                    # Optional: convergence tolerance

Geo (Geometric)
~~~~~~~~~~~~~~~

Best for: High-quality alignment, moderate number of patches

.. code-block:: python

   aligner = get_aligner("geo",
       method="orthogonal",     # or "euclidean"
       use_scale=True,          # Enable scale optimization
       num_epochs=10,           # Optimization epochs
       learning_rate=0.01,      # Learning rate for manifold optimization
       verbose=True             # Show optimization progress
   )

Tips and Best Practices
------------------------

1. **Choosing Embedding Dimensions**: 
   - Start with 64-128 for most graphs
   - Higher dimensions (256+) for very large or complex graphs
   - Lower dimensions (32-64) for faster computation

2. **Patch Configuration**:
   - **num_patches**: 8-12 patches work well for most graphs
   - **min_overlap/target_overlap**: 10-30 nodes for good alignment
   - **clustering_method**: "metis" is usually best

3. **Base Methods**:
   - **svd**: Fastest, good baseline performance
   - **vgae**: Best quality, requires more computation
   - **dgi**: Good for self-supervised learning

4. **Aligner Selection**:
   - **L2G**: Use for >10 patches or when speed is important
   - **Geo**: Use for ≤8 patches when quality is most important

5. **Hierarchical vs Patched**:
   - **Patched**: Better for uniform graph structure
   - **Hierarchical**: Better for graphs with natural hierarchy

6. **Performance Tips**:
   - Use ``verbose=True`` during development to monitor progress
   - Start with small configurations, then scale up
   - Use SVD base method for fast prototyping

Complete Example
----------------

Here's a complete workflow comparing different embedding approaches:

.. code-block:: python

   import numpy as np
   from l2gx.datasets import get_dataset
   from l2gx.embedding import get_embedding
   from l2gx.align import get_aligner
   from l2gx.graphs import TGraph
   
   # Load dataset
   dataset = get_dataset("Cora")
   data = TGraph.from_tg(dataset.to("torch-geometric"))
   
   results = {}
   
   # 1. Simple VGAE embedding
   simple_embedder = get_embedding("vgae", embedding_dim=128, epochs=100)
   results['simple'] = simple_embedder.fit_transform(data.to_tg())
   
   # 2. Patched embedding with L2G alignment
   l2g_aligner = get_aligner("l2g")
   l2g_aligner.randomized_method = "randomized"
   
   patched_embedder = get_embedding(
       "patched", embedding_dim=128, aligner=l2g_aligner,
       num_patches=10, base_method="vgae", epochs=100
   )
   results['patched_l2g'] = patched_embedder.fit_transform(data.to_tg())
   
   # 3. Patched embedding with Geo alignment
   geo_aligner = get_aligner("geo", method="orthogonal", use_scale=True)
   
   patched_geo_embedder = get_embedding(
       "patched", embedding_dim=128, aligner=geo_aligner,
       num_patches=6, base_method="svd"  # SVD is fast for Geo
   )
   results['patched_geo'] = patched_geo_embedder.fit_transform(data.to_tg())
   
   # 4. Hierarchical embedding
   hier_aligner = get_aligner("l2g")  # Currently uses Procrustes regardless
   
   hier_embedder = get_embedding(
       "hierarchical", embedding_dim=128, aligner=hier_aligner,
       max_patch_size=400, branching_factor=3, base_method="svd"
   )
   results['hierarchical'] = hier_embedder.fit_transform(data.to_tg())
   
   # Compare results
   for method, embedding in results.items():
       print(f"{method}: {embedding.shape}, mean_norm: {np.mean(np.linalg.norm(embedding, axis=1)):.3f}")

This will output the embedding shapes and mean norms for comparison across methods.