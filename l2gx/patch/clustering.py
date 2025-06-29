"""
Graph Clustering Algorithms for Patch Generation (Compatibility Layer)

This module now serves as a compatibility layer that imports all clustering algorithms
from the organized clustering subdirectory. The algorithms have been split into separate
modules for better organization and maintainability.

For new code, consider importing directly from the clustering submodules:
    ```python
    from l2gx.patch.clustering.fennel import fennel_clustering
    from l2gx.patch.clustering.louvain import louvain_clustering
    ```

This compatibility layer ensures existing code continues to work:
    ```python
    from l2gx.patch.clustering import fennel_clustering, louvain_clustering
    ```

Available Algorithms:

1. **Fennel Clustering** - Single-pass streaming algorithm with load balancing
2. **Louvain Clustering** - Modularity-based community detection
3. **METIS Clustering** - Multi-level graph partitioning
4. **Spread Clustering** - Degree-based spreading algorithm
5. **Hierarchical Clustering** - Multi-level clustering with size constraints

For detailed documentation, see the individual algorithm modules in the clustering/ subdirectory.
"""

import warnings

# Import everything from the new organized clustering package
from .clustering import *

# Issue a deprecation warning for direct imports from this module
def _deprecated_import_warning():
    warnings.warn(
        "Importing clustering algorithms directly from l2gx.patch.clustering is deprecated. "
        "Please import from the specific algorithm modules, e.g., "
        "from l2gx.patch.clustering.fennel import fennel_clustering",
        DeprecationWarning,
        stacklevel=3
    )

# Override some imports to add deprecation warnings
_original_fennel = fennel_clustering
_original_louvain = louvain_clustering
_original_metis = metis_clustering
_original_spread = spread_clustering

def fennel_clustering(*args, **kwargs):
    """Deprecated: Use l2gx.patch.clustering.fennel.fennel_clustering instead"""
    _deprecated_import_warning()
    return _original_fennel(*args, **kwargs)

def louvain_clustering(*args, **kwargs):
    """Deprecated: Use l2gx.patch.clustering.louvain.louvain_clustering instead"""
    _deprecated_import_warning()
    return _original_louvain(*args, **kwargs)

def metis_clustering(*args, **kwargs):
    """Deprecated: Use l2gx.patch.clustering.metis.metis_clustering instead"""
    _deprecated_import_warning()
    return _original_metis(*args, **kwargs)

def spread_clustering(*args, **kwargs):
    """Deprecated: Use l2gx.patch.clustering.spread.spread_clustering instead"""
    _deprecated_import_warning()
    return _original_spread(*args, **kwargs)

# Keep the existing legacy imports for backward compatibility without warnings
# These are used internally and don't need deprecation warnings
try:
    from .clustering.fennel import _fennel_clustering
except ImportError:
    # Fallback for the original implementation
    from raphtory import Graph
    import numpy as np
    import numba
    import torch
    from tqdm import tqdm
    from l2gx.utils import tqdm_close
    
    @numba.njit
    def _fennel_clustering(
        edge_index: np.ndarray,
        adj_index: np.ndarray,
        num_nodes: int,
        num_clusters: int,
        load_limit: float = 1.1,
        alpha: float | None = None,
        gamma: float = 1.5,
        num_iters: int = 1,
        clusters=np.empty(0, dtype=np.int64),
    ):
        """Legacy _fennel_clustering implementation for backward compatibility"""
        # This is a simplified version - the full implementation is in clustering/fennel.py
        num_edges = edge_index.shape[1]
        if alpha is None:
            alpha = num_edges * (num_clusters ** (gamma - 1)) / (num_nodes**gamma)
        
        partition_sizes = np.zeros(num_clusters, dtype=np.int64)
        if clusters.size == 0:
            clusters = np.full((num_nodes,), -1, dtype=np.int64)
        else:
            clusters = np.copy(clusters)
            for index in clusters:
                partition_sizes[index] += 1

        load_limit *= num_nodes / num_clusters
        assert alpha
        deltas = -alpha * gamma * (partition_sizes ** (gamma - 1))

        for it in range(num_iters):
            not_converged = 0
            for i in range(num_nodes):
                cluster_indices = np.empty(
                    (adj_index[i + 1] - adj_index[i],), dtype=np.int64
                )
                for ni, index in enumerate(range(adj_index[i], adj_index[i + 1])):
                    cluster_indices[ni] = clusters[edge_index[1, index]]
                old_cluster = clusters[i]
                if old_cluster >= 0:
                    partition_sizes[old_cluster] -= 1
                cluster_indices = cluster_indices[cluster_indices >= 0]

                if cluster_indices.size > 0:
                    c_size = np.zeros(num_clusters, dtype=np.int64)
                    for index in cluster_indices:
                        c_size[index] += 1
                    ind = np.argmax(deltas + c_size)
                else:
                    ind = np.argmax(deltas)
                clusters[i] = ind
                partition_sizes[ind] += 1
                if partition_sizes[ind] == load_limit:
                    deltas[ind] = -np.inf
                else:
                    deltas[ind] = -alpha * gamma * (partition_sizes[ind] ** (gamma - 1))
                not_converged += ind != old_cluster

            if not_converged == 0:
                break

        return clusters

# Import the Partition class for backward compatibility
try:
    from .clustering.utils import Partition
except ImportError:
    # Fallback Partition implementation
    from typing import Sequence
    import torch
    
    class Partition(Sequence):
        """Fallback Partition class for backward compatibility"""
        def __init__(self, partition_tensor):
            partition_tensor = torch.as_tensor(partition_tensor)
            counts = torch.bincount(partition_tensor)
            self.num_parts = len(counts)
            self.nodes = torch.argsort(partition_tensor)
            self.part_index = torch.zeros(self.num_parts + 1, dtype=torch.long)
            self.part_index[1:] = torch.cumsum(counts, dim=0)

        def __getitem__(self, item):
            return self.nodes[self.part_index[item] : self.part_index[item + 1]]

        def __len__(self):
            return self.num_parts