"""
Graph embedding method implementations.

This package contains concrete implementations of various graph embedding methods,
all following the GraphEmbedding interface defined in the base module.
"""

# Import all embedding methods to trigger registration
from .gae_embedding import GAEEmbedding, VGAEEmbedding
from .svd_embedding import SVDEmbedding
from .graphsage_embedding import GraphSAGEEmbedding
from .dgi_embedding import DGIEmbedding
from .hierarchical_embedding import HierarchicalEmbedding
from .patched_embedding import PatchedEmbedding

__all__ = [
    "GAEEmbedding",
    "VGAEEmbedding",
    "SVDEmbedding",
    "GraphSAGEEmbedding",
    "DGIEmbedding",
    "HierarchicalEmbedding",
    "PatchedEmbedding",
]
