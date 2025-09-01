"""
GAT (Graph Attention Network) embedding method.

This module provides GAT embedding implementation that follows the standard 
GraphEmbedding interface. GAT uses attention mechanisms to weight the importance
of neighboring nodes during aggregation, allowing the model to learn which 
connections are most important for the task at hand.
"""

from typing import Union, Literal, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import negative_sampling
import networkx as nx
import scipy.sparse as sp

from ..base import InductiveGraphEmbedding
from ..registry import register_embedding
from ..utils import convert_graph_format


class GATModel(nn.Module):
    """
    Graph Attention Network (GAT) model implementation.
    
    This implements the GAT architecture which uses multi-head attention
    mechanisms to aggregate information from neighboring nodes, learning
    the importance of different connections.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.6,
        attention_dropout: float = 0.6,
        negative_slope: float = 0.2,
        pooling: Literal["mean", "max", "add"] = "mean",
        concat_heads: bool = True,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        
        # Pooling function for graph-level representations
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        elif pooling == "add":
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        # Create GAT layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(
            input_dim, 
            hidden_dim,
            heads=num_heads,
            dropout=attention_dropout,
            negative_slope=negative_slope,
            concat=concat_heads
        ))
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            # Account for concatenated heads if concat=True
            in_dim = hidden_dim * num_heads if concat_heads else hidden_dim
            self.convs.append(GATConv(
                in_dim,
                hidden_dim,
                heads=num_heads,
                dropout=attention_dropout,
                negative_slope=negative_slope,
                concat=concat_heads
            ))
        
        # Final layer (typically doesn't concatenate heads)
        if num_layers > 1:
            in_dim = hidden_dim * num_heads if concat_heads else hidden_dim
            # Final layer: use single head or average multi-heads
            self.convs.append(GATConv(
                in_dim,
                output_dim,
                heads=1,  # Single head for final layer
                dropout=attention_dropout,
                negative_slope=negative_slope,
                concat=False
            ))

    def forward(self, x, edge_index, batch=None, return_attention_weights=False):
        """
        Forward pass through GAT layers.
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch indices for graph-level pooling
            return_attention_weights: Whether to return attention weights
        
        Returns:
            Node embeddings or graph-level embeddings (if batch is provided)
        """
        attention_weights = []
        
        # Apply GAT layers
        for i, conv in enumerate(self.convs):
            # Apply dropout to input features (except for first layer)
            if i > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            if return_attention_weights:
                x, (edge_index_with_weights, alpha) = conv(
                    x, edge_index, return_attention_weights=True
                )
                attention_weights.append((edge_index_with_weights, alpha))
            else:
                x = conv(x, edge_index)
            
            # Apply activation (except for last layer)
            if i < self.num_layers - 1:
                x = F.elu(x)
        
        # Apply pooling if batch indices are provided (for graph-level tasks)
        if batch is not None:
            x = self.pool(x, batch)
        
        if return_attention_weights:
            return x, attention_weights
        return x


@register_embedding("gat")
class GATEmbedding(InductiveGraphEmbedding):
    """
    Graph Attention Network (GAT) embedding implementation.
    
    GAT uses attention mechanisms to learn the importance of different
    edges in the graph, making it particularly effective for graphs where
    some connections are more important than others.
    
    Parameters
    ----------
    embedding_dim : int, default=128
        Dimension of the output embeddings
    hidden_dim : int, default=256
        Dimension of hidden layers
    num_layers : int, default=2
        Number of GAT layers
    num_heads : int, default=8
        Number of attention heads per layer
    dropout : float, default=0.6
        Dropout rate for features
    attention_dropout : float, default=0.6
        Dropout rate for attention weights
    negative_slope : float, default=0.2
        Negative slope for LeakyReLU in attention mechanism
    learning_rate : float, default=0.005
        Learning rate for optimization
    epochs : int, default=200
        Number of training epochs
    negative_samples : int, default=5
        Number of negative samples for unsupervised training
    pooling : str, default="mean"
        Pooling method for graph-level representations ("mean", "max", or "add")
    concat_heads : bool, default=True
        Whether to concatenate attention heads
    device : str, default="auto"
        Device to use ("cpu", "cuda", or "auto")
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.6,
        attention_dropout: float = 0.6,
        negative_slope: float = 0.2,
        learning_rate: float = 0.005,
        epochs: int = 200,
        negative_samples: int = 5,
        pooling: Literal["mean", "max", "add"] = "mean",
        concat_heads: bool = True,
        device: str = "auto",
    ):
        super().__init__(embedding_dim=embedding_dim)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.negative_slope = negative_slope
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.negative_samples = negative_samples
        self.pooling = pooling
        self.concat_heads = concat_heads
        
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.optimizer = None

    def _initialize_model(self, input_dim: int):
        """Initialize the GAT model and optimizer."""
        self.model = GATModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.embedding_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            negative_slope=self.negative_slope,
            pooling=self.pooling,
            concat_heads=self.concat_heads,
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

    def _prepare_features(self, data: Data) -> torch.Tensor:
        """Prepare node features, creating dummy features if needed."""
        if data.x is not None:
            return data.x
        else:
            # Create one-hot encoded features based on degree
            num_nodes = data.num_nodes
            degrees = torch.zeros(num_nodes, dtype=torch.long)
            
            for i in range(data.edge_index.size(1)):
                src = data.edge_index[0, i]
                degrees[src] += 1
            
            max_degree = min(degrees.max().item() + 1, 100)
            x = torch.zeros(num_nodes, max_degree)
            for i in range(num_nodes):
                deg = min(degrees[i].item(), max_degree - 1)
                x[i, deg] = 1.0
            
            return x

    def _train_epoch(self, data: Data, x: torch.Tensor) -> float:
        """Train for one epoch using link prediction objective."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Get node embeddings
        z = self.model(x, data.edge_index)
        
        # Positive edges (existing edges)
        pos_edge_index = data.edge_index
        
        # Negative edges (non-existing edges)
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1) * self.negative_samples,
        )
        
        # Compute scores for positive and negative edges
        pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
        neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
        
        # Binary cross-entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )
        loss = pos_loss + neg_loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def fit(self, graph: Union[nx.Graph, Data, sp.spmatrix], verbose: bool = True) -> "GATEmbedding":
        """
        Fit the GAT model to the graph.
        
        Parameters
        ----------
        graph : NetworkX graph, PyTorch Geometric Data, or scipy sparse matrix
            The input graph to embed
        verbose : bool, default=True
            Whether to print training progress
        
        Returns
        -------
        self : GATEmbedding
            The fitted embedding model
        """
        # Convert to PyTorch Geometric format
        data = convert_graph_format(graph, "torch_geometric")
        data = data.to(self.device)
        
        # Prepare features
        x = self._prepare_features(data).to(self.device)
        
        # Initialize model
        self._initialize_model(x.size(1))
        
        # Training loop
        for epoch in range(self.epochs):
            loss = self._train_epoch(data, x)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")
        
        return self

    def transform(self, graph: Optional[Union[nx.Graph, Data, sp.spmatrix]] = None) -> np.ndarray:
        """
        Transform the graph to get embeddings.
        
        Parameters
        ----------
        graph : NetworkX graph, PyTorch Geometric Data, or scipy sparse matrix, optional
            The graph to transform. If None, uses the training graph.
        
        Returns
        -------
        embeddings : np.ndarray
            Node embeddings of shape (n_nodes, embedding_dim)
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Convert to PyTorch Geometric format
        data = convert_graph_format(graph, "torch_geometric") if graph is not None else self._last_data
        data = data.to(self.device)
        
        # Prepare features
        x = self._prepare_features(data).to(self.device)
        
        # Get embeddings
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(x, data.edge_index)
        
        return embeddings.cpu().numpy()

    def fit_transform(self, graph: Union[nx.Graph, Data, sp.spmatrix], verbose: bool = True) -> np.ndarray:
        """
        Fit the model and return embeddings.
        
        Parameters
        ----------
        graph : NetworkX graph, PyTorch Geometric Data, or scipy sparse matrix
            The input graph to embed
        verbose : bool, default=True
            Whether to print training progress
        
        Returns
        -------
        embeddings : np.ndarray
            Node embeddings of shape (n_nodes, embedding_dim)
        """
        # Store data for transform
        self._last_data = convert_graph_format(graph, "torch_geometric")
        
        self.fit(graph, verbose=verbose)
        return self.transform()

    def transform_new_nodes(
        self,
        graph: Union[nx.Graph, Data, sp.spmatrix],
        new_node_ids: list
    ) -> np.ndarray:
        """
        Generate embeddings for new nodes in an inductive manner.
        
        Parameters
        ----------
        graph : NetworkX graph, PyTorch Geometric Data, or scipy sparse matrix
            The graph containing both old and new nodes
        new_node_ids : list
            List of new node IDs to generate embeddings for
        
        Returns
        -------
        embeddings : np.ndarray
            Embeddings for the new nodes of shape (len(new_node_ids), embedding_dim)
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Convert to PyTorch Geometric format
        data = convert_graph_format(graph, "torch_geometric")
        data = data.to(self.device)
        
        # Prepare features
        x = self._prepare_features(data).to(self.device)
        
        # Get embeddings for all nodes
        self.model.eval()
        with torch.no_grad():
            all_embeddings = self.model(x, data.edge_index)
        
        # Extract embeddings for new nodes
        new_node_ids = torch.LongTensor(new_node_ids).to(self.device)
        new_embeddings = all_embeddings[new_node_ids]
        
        return new_embeddings.cpu().numpy()

    def get_attention_weights(self, graph: Union[nx.Graph, Data, sp.spmatrix]) -> list:
        """
        Get attention weights for each layer.
        
        Parameters
        ----------
        graph : NetworkX graph, PyTorch Geometric Data, or scipy sparse matrix
            The input graph
        
        Returns
        -------
        attention_weights : list
            List of (edge_index, attention_weights) tuples for each layer
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Convert to PyTorch Geometric format
        data = convert_graph_format(graph, "torch_geometric")
        data = data.to(self.device)
        
        # Prepare features
        x = self._prepare_features(data).to(self.device)
        
        # Get embeddings with attention weights
        self.model.eval()
        with torch.no_grad():
            _, attention_weights = self.model(
                x, data.edge_index, return_attention_weights=True
            )
        
        return attention_weights