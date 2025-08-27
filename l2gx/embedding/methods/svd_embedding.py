"""
SVD-based embedding methods.

This module provides SVD-based graph embedding implementations that follow
the standard GraphEmbedding interface.
"""

from typing import Union, Optional, Literal
import numpy as np
import scipy.sparse as sp
import networkx as nx
from torch_geometric.data import Data

from ..base import TransductiveGraphEmbedding
from ..registry import register_embedding

# SVD functions implemented directly here
from ..utils import convert_graph_format


def _simplified_svd(matrix, k, random_state=None):
    """Simplified SVD using scipy for embedding purposes."""
    try:
        from scipy.sparse.linalg import svds

        U, s, Vt = svds(matrix, k=k, random_state=random_state)
        return U, s, Vt
    except ImportError:
        # Fallback to numpy SVD for dense matrices
        U, s, Vt = np.linalg.svd(
            matrix.toarray() if hasattr(matrix, "toarray") else matrix
        )
        return U[:, :k], s[:k], Vt[:k, :]


@register_embedding("svd", aliases=["singular_value_decomposition"])
class SVDEmbedding(TransductiveGraphEmbedding):
    """
    SVD-based graph embedding method.

    Computes node embeddings using Singular Value Decomposition of the
    adjacency matrix or its normalized variants.
    """

    def __init__(
        self,
        embedding_dim: int,
        matrix_type: Literal["adjacency", "laplacian", "normalized"] = "adjacency",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize SVD embedding method.

        Args:
            embedding_dim: Output embedding dimensionality
            matrix_type: Type of matrix to decompose ('adjacency', 'laplacian', 'normalized')
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        super().__init__(embedding_dim, **kwargs)
        self.matrix_type = matrix_type
        self.random_state = random_state

        self._embeddings = None
        self._node_indices = None

    def fit(
        self, graph: Union[Data, nx.Graph, sp.spmatrix, np.ndarray], verbose: bool = False
    ) -> "SVDEmbedding":
        """
        Fit the SVD embedding to the graph.

        Args:
            graph: Input graph in supported format
            verbose: Unused, SVD doesn't have verbose output

        Returns:
            self: Fitted embedding instance
        """
        self._validate_graph_format(graph)

        # Convert to sparse matrix format
        adj_matrix = convert_graph_format(graph, target_format="scipy_sparse")

        # Regular SVD embedding (simplified, no bipartite support for now)
        if self.matrix_type == "adjacency":
            matrix = adj_matrix
        elif self.matrix_type == "laplacian":
            # Compute Laplacian matrix
            degree = np.array(adj_matrix.sum(axis=1)).flatten()
            D = sp.diags(degree)
            matrix = D - adj_matrix
        elif self.matrix_type == "normalized":
            # Compute normalized Laplacian
            degree = np.array(adj_matrix.sum(axis=1)).flatten()
            degree_inv_sqrt = np.power(
                degree, -0.5, out=np.zeros_like(degree), where=(degree != 0)
            )
            D_inv_sqrt = sp.diags(degree_inv_sqrt)
            matrix = sp.eye(adj_matrix.shape[0]) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt
        else:
            raise ValueError(f"Unknown matrix_type: {self.matrix_type}")

        # Perform SVD
        U, s, _ = _simplified_svd(
            matrix, k=self.embedding_dim, random_state=self.random_state
        )

        if self.matrix_type == "adjacency":
            # Use left singular vectors weighted by singular values
            self._embeddings = U * np.sqrt(s)
        else:
            # For Laplacian matrices, use left singular vectors
            self._embeddings = U

        self._node_indices = np.arange(adj_matrix.shape[0])

        self.is_fitted = True
        return self

    def transform(
        self, graph: Union[Data, nx.Graph, sp.spmatrix, np.ndarray]
    ) -> np.ndarray:
        """
        Return the computed node embeddings.

        Args:
            graph: Input graph (should be the same as used for fitting)

        Returns:
            Node embeddings of shape (num_nodes, embedding_dim)
        """
        del graph  # Parameter not used for fitted SVD model
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform()")

        return self._embeddings.copy()

    def get_embedding_matrix(self) -> np.ndarray:
        """
        Get the embedding matrix directly.

        Returns:
            Embedding matrix of shape (num_nodes, embedding_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before getting embeddings")

        return self._embeddings.copy()

    def get_node_indices(self) -> np.ndarray:
        """
        Get the node indices corresponding to the embeddings.

        Returns:
            Array of node indices
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before getting node indices")

        return self._node_indices.copy()
