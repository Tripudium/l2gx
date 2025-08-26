"""
Graph embedding module for L2GX.

This module provides a unified interface for various graph embedding methods,
including Graph Auto-Encoders (GAE/VGAE) and SVD-based methods.

Main classes:
- GraphEmbedding: Abstract base class for all embedding methods
- GAEEmbedding: Graph Auto-Encoder implementation
- VGAEEmbedding: Variational Graph Auto-Encoder implementation
- SVDEmbedding: SVD-based embedding implementation
- GraphSAGEEmbedding: GraphSAGE inductive embedding implementation
- DGIEmbedding: Deep Graph Infomax self-supervised embedding implementation

Registry functions:
- get_embedding: Create embedding instances by name
- list_embeddings: list available embedding methods
"""

from .base import GraphEmbedding, InductiveGraphEmbedding, TransductiveGraphEmbedding
from .registry import register_embedding, get_embedding, list_embeddings

from .methods import (
    GAEEmbedding,
    VGAEEmbedding,
    SVDEmbedding,
    GraphSAGEEmbedding,
    DGIEmbedding,
)
from .patched_embedding import PatchedEmbedding

# from .supervised_patch_embedding import SupervisedPatchedEmbedding  # Temporarily disabled
from .train import train_gae
from .utils import convert_graph_format

__all__ = [
    # Unified embedding interface
    "GraphEmbedding",
    "InductiveGraphEmbedding",
    "TransductiveGraphEmbedding",
    "GAEEmbedding",
    "VGAEEmbedding",
    "SVDEmbedding",
    "GraphSAGEEmbedding",
    "DGIEmbedding",
    "PatchedEmbedding",
    # "SupervisedPatchedEmbedding",  # Temporarily disabled
    "get_embedding",
    "list_embeddings",
    "register_embedding",
    "convert_graph_format",
    "train_gae",
]
