"""
Graph Auto-Encoder embedding methods.

This module provides GAE and VGAE embedding implementations that follow
the standard GraphEmbedding interface.
"""


import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GAE, VGAE, GCNConv

from ..base import TransductiveGraphEmbedding
from ..registry import register_embedding
from ..train import train_gae
from ..utils import convert_graph_format


class GCNEncoder(nn.Module):
    """
    GCN encoder for GAE.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Match old implementation parameters
        self.conv1 = GCNConv(
            input_dim,
            hidden_dim,
            cached=True,
            bias=True,
            add_self_loops=True,
            normalize=True,
        )
        self.conv2 = GCNConv(
            hidden_dim,
            output_dim,
            cached=True,
            bias=True,
            add_self_loops=True,
            normalize=True,
        )

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(nn.Module):
    """
    Variational GCN encoder for VGAE.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Match old implementation parameters
        self.conv1 = GCNConv(
            input_dim,
            hidden_dim,
            cached=True,
            bias=True,
            add_self_loops=True,
            normalize=True,
        )
        self.conv_mu = GCNConv(
            hidden_dim,
            output_dim,
            cached=True,
            bias=True,
            add_self_loops=True,
            normalize=True,
        )
        self.conv_logstd = GCNConv(
            hidden_dim,
            output_dim,
            cached=True,
            bias=True,
            add_self_loops=True,
            normalize=True,
        )

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


@register_embedding("gae", aliases=["graph_autoencoder"])
class GAEEmbedding(TransductiveGraphEmbedding):
    """
    Graph Auto-Encoder embedding method.

    Implements a Graph Auto-Encoder that learns node embeddings by reconstructing
    the graph's adjacency matrix through an encoder-decoder architecture.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        epochs: int = 1000,
        patience: int | None = None,
        distance_decoder: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize GAE embedding method.

        Args:
            embedding_dim: Output embedding dimensionality
            hidden_dim: Hidden layer dimensionality in encoder
            learning_rate: Learning rate for training
            weight_decay: Weight decay for optimizer
            epochs: Number of training epochs
            patience: Early stopping patience (None to disable)
            distance_decoder: Use distance-based decoder instead of inner product
            device: Device to run computations on ('cpu' or 'cuda')
            **kwargs: Additional parameters
        """
        super().__init__(embedding_dim, **kwargs)
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.distance_decoder = distance_decoder
        self.device = device

        self._model = None
        self._data = None

    def fit(
        self, graph: Data | nx.Graph | sp.spmatrix | np.ndarray, verbose: bool = False
    ) -> "GAEEmbedding":
        """
        Fit the GAE model to the graph.

        Args:
            graph: Input graph in supported format

        Returns:
            self: Fitted embedding instance
        """
        self._validate_graph_format(graph)

        # Convert to PyTorch Geometric format
        self._data = convert_graph_format(graph, target_format="torch_geometric")

        # Initialize model
        num_features = (
            self._data.x.size(1) if self._data.x is not None else self._data.num_nodes
        )
        if self._data.x is None:
            self._data.x = torch.eye(self._data.num_nodes, dtype=torch.float)

        encoder = GCNEncoder(num_features, self.hidden_dim, self.embedding_dim)
        self._model = GAE(encoder).to(self.device)

        # Move data to device
        self._data = self._data.to(self.device)

        # Train the model
        _, self.training_history = train_gae(
            model=self._model,
            data=self._data,
            epochs=self.epochs,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            patience=self.patience,
            variational=False,
            verbose=verbose,
        )

        self.is_fitted = True
        return self

    def transform(
        self, graph: Data | nx.Graph | sp.spmatrix | np.ndarray
    ) -> np.ndarray:
        """
        Generate node embeddings using the fitted GAE model.

        Args:
            graph: Input graph (should be the same as used for fitting)

        Returns:
            Node embeddings of shape (num_nodes, embedding_dim)
        """
        del graph  # Parameter not needed for fitted model
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform()")

        self._model.eval()
        with torch.no_grad():
            embeddings = self._model.encode(self._data.x, self._data.edge_index)

        return embeddings.cpu().numpy()


@register_embedding(
    "vgae", aliases=["variational_gae", "variational_graph_autoencoder"]
)
class VGAEEmbedding(TransductiveGraphEmbedding):
    """
    Variational Graph Auto-Encoder embedding method.

    Implements a Variational Graph Auto-Encoder that learns probabilistic node
    embeddings by reconstructing the graph structure through a variational
    encoder-decoder architecture.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        epochs: int = 1000,
        patience: int | None = None,
        distance_decoder: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize VGAE embedding method.

        Args:
            embedding_dim: Output embedding dimensionality
            hidden_dim: Hidden layer dimensionality in encoder
            learning_rate: Learning rate for training
            weight_decay: Weight decay for optimizer
            epochs: Number of training epochs
            patience: Early stopping patience (None to disable)
            distance_decoder: Use distance-based decoder instead of inner product
            device: Device to run computations on ('cpu' or 'cuda')
            **kwargs: Additional parameters
        """
        super().__init__(embedding_dim, **kwargs)
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.distance_decoder = distance_decoder
        self.device = device

        self._model = None
        self._data = None

    def fit(
        self, graph: Data | nx.Graph | sp.spmatrix | np.ndarray, verbose: bool = False
    ) -> "VGAEEmbedding":
        """
        Fit the VGAE model to the graph.

        Args:
            graph: Input graph in supported format

        Returns:
            self: Fitted embedding instance
        """
        self._validate_graph_format(graph)

        # Convert to PyTorch Geometric format
        self._data = convert_graph_format(graph, target_format="torch_geometric")

        # Initialize model
        num_features = (
            self._data.x.size(1) if self._data.x is not None else self._data.num_nodes
        )
        if self._data.x is None:
            self._data.x = torch.eye(self._data.num_nodes, dtype=torch.float)

        encoder = VariationalGCNEncoder(
            num_features, self.hidden_dim, self.embedding_dim
        )
        self._model = VGAE(encoder).to(self.device)

        # Move data to device
        self._data = self._data.to(self.device)

        # Train the model
        _, self.training_history = train_gae(
            model=self._model,
            data=self._data,
            epochs=self.epochs,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            patience=self.patience,
            variational=True,
            verbose=verbose
        )

        self.is_fitted = True
        return self

    def transform(
        self, graph: Data | nx.Graph | sp.spmatrix | np.ndarray
    ) -> np.ndarray:
        """
        Generate node embeddings using the fitted VGAE model.

        Args:
            graph: Input graph (should be the same as used for fitting)

        Returns:
            Node embeddings of shape (num_nodes, embedding_dim)
        """
        del graph  # Parameter not needed for fitted model
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform()")

        self._model.eval()
        with torch.no_grad():
            # For VGAE, use the mean of the variational distribution
            embeddings = self._model.encode(self._data.x, self._data.edge_index)
            if isinstance(embeddings, tuple):  # VGAE returns (mu, logvar)
                embeddings = embeddings[0]  # Use mean

        return embeddings.cpu().numpy()
