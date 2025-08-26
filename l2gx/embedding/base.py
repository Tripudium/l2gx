"""
Abstract base class for graph embedding methods.

This module provides the core GraphEmbedding interface that all embedding methods
should implement, enabling consistent usage across different embedding techniques.
"""

from abc import ABC, abstractmethod
# Built-in types used directly

import networkx as nx
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data


class GraphEmbedding(ABC):
    """
    Abstract base class for graph embedding methods.

    This class defines the standard interface that all graph embedding methods
    should implement. It provides a consistent API for fitting embeddings to
    graphs and transforming graphs into node embeddings.
    """

    def __init__(self, embedding_dim: int, **kwargs):
        """
        Initialize the graph embedding method.

        Args:
            embedding_dim: Target dimensionality of node embeddings
            **kwargs: Method-specific parameters
        """
        self.embedding_dim = embedding_dim
        self.is_fitted = False
        self._parameters = kwargs
        self.training_history = None

    @abstractmethod
    def fit(
        self, graph: Data | nx.Graph | sp.spmatrix | np.ndarray, verbose: bool = False
    ) -> "GraphEmbedding":
        """
        Learn embedding parameters from graph structure.

        Args:
            graph: Input graph in supported format (PyTorch Geometric Data,
                   NetworkX Graph, scipy sparse matrix, or numpy adjacency matrix)

        Returns:
            self: Fitted embedding instance for method chaining
        """
        pass

    @abstractmethod
    def transform(
        self, graph: Data | nx.Graph | sp.spmatrix | np.ndarray
    ) -> np.ndarray:
        """
        Generate node embeddings for the given graph.

        Args:
            graph: Input graph in supported format

        Returns:
            Node embeddings as numpy array of shape (num_nodes, embedding_dim)

        Raises:
            RuntimeError: If called before fit() for methods that require training
        """
        pass

    def fit_transform(
        self, graph: Data | nx.Graph | sp.spmatrix | np.ndarray, verbose: bool = False
    ) -> np.ndarray:
        """
        Fit the embedding method and transform the graph in one step.

        Args:
            graph: Input graph in supported format

        Returns:
            Node embeddings as numpy array of shape (num_nodes, embedding_dim)
        """
        return self.fit(graph, verbose=verbose).transform(graph)

    def get_params(self) -> dict[str, any]:
        """
        Get parameters for this embedding method.

        Returns:
            Dictionary of parameter names and values
        """
        params = {"embedding_dim": self.embedding_dim}
        params.update(self._parameters)
        return params

    def set_params(self, **params) -> "GraphEmbedding":
        """
        set parameters for this embedding method.

        Args:
            **params: Parameter names and values to set

        Returns:
            self: For method chaining
        """
        if "embedding_dim" in params:
            self.embedding_dim = params.pop("embedding_dim")

        self._parameters.update(params)
        self.is_fitted = False  # Reset fitted state when parameters change
        return self

    @staticmethod
    def _validate_graph_format(
        graph: Data | nx.Graph | sp.spmatrix | np.ndarray,
    ) -> None:
        """
        Validate that the input graph is in a supported format.

        Args:
            graph: Input graph to validate

        Raises:
            TypeError: If graph format is not supported
        """
        supported_types = (Data, nx.Graph, sp.spmatrix, np.ndarray)
        if not isinstance(graph, supported_types):
            raise TypeError(
                f"Graph must be one of {[t.__name__ for t in supported_types]}, "
                f"got {type(graph).__name__}"
            )

    @staticmethod
    def _get_num_nodes(graph: Data | nx.Graph | sp.spmatrix | np.ndarray) -> int:
        """
        Get the number of nodes in the graph regardless of format.

        Args:
            graph: Input graph

        Returns:
            Number of nodes in the graph
        """
        if isinstance(graph, Data):
            return graph.num_nodes
        if isinstance(graph, nx.Graph):
            return graph.number_of_nodes()
        if isinstance(graph, sp.spmatrix):
            return graph.shape[0]
        if isinstance(graph, np.ndarray):
            return graph.shape[0]

        raise TypeError(f"Unsupported graph type: {type(graph)}")

    def get_training_history(self) -> dict[str, list] | None:
        """
        Get the training history recorded during fit().

        Returns:
            Dictionary containing training metrics over epochs, or None if not available.
            Typically contains 'epochs' and 'losses' keys with lists of values.
        """
        return self.training_history

    def __repr__(self) -> str:
        """String representation of the embedding method."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params_str})"


class InductiveGraphEmbedding(GraphEmbedding):
    """
    Base class for inductive graph embedding methods.

    Inductive methods can generate embeddings for new nodes without retraining.
    This is useful for dynamic graphs or when working with large graphs where
    new nodes are frequently added.
    """

    @abstractmethod
    def transform_new_nodes(
        self,
        graph: Data | nx.Graph | sp.spmatrix | np.ndarray,
        new_node_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Generate embeddings for new nodes without retraining.

        Args:
            graph: Input graph containing both old and new nodes
            new_node_indices: Indices of the new nodes to embed

        Returns:
            Embeddings for the new nodes of shape (len(new_node_indices), embedding_dim)
        """


class TransductiveGraphEmbedding(GraphEmbedding):
    """
    Base class for transductive graph embedding methods.

    Transductive methods learn embeddings for a fixed set of nodes and
    require retraining to handle new nodes.
    """

    def transform_new_nodes(
        self,
        graph: Data | nx.Graph | sp.spmatrix | np.ndarray,
        new_node_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Transductive methods cannot embed new nodes without retraining.

        Raises:
            NotImplementedError: Always, as transductive methods require retraining
        """
        del graph, new_node_indices  # Mark parameters as intentionally unused
        raise NotImplementedError(
            "Transductive embedding methods cannot embed new nodes without retraining. "
            "Use fit_transform on the full graph including new nodes."
        )
