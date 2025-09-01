"""
GIN (Graph Isomorphism Network) embedding method.

This module provides GIN embedding implementation that follows the standard 
GraphEmbedding interface. GIN is designed to be as powerful as the 
Weisfeiler-Lehman test for graph isomorphism and is particularly effective 
for graph classification tasks.
"""

from typing import Union, Literal, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import negative_sampling
import networkx as nx
import scipy.sparse as sp

from ..base import InductiveGraphEmbedding
from ..registry import register_embedding
from ..utils import convert_graph_format


class GINModel(nn.Module):
    """
    Graph Isomorphism Network (GIN) model implementation.
    
    This implements the GIN architecture which uses learnable aggregation
    functions and can theoretically distinguish any graphs that the 
    Weisfeiler-Lehman test can distinguish.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        eps: float = 0.0,
        train_eps: bool = True,
        pooling: Literal["mean", "max", "add"] = "mean",
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.eps = eps
        self.train_eps = train_eps
        
        # Pooling function
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        elif pooling == "add":
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        # Create GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Layer dimensions
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(num_layers):
            # MLP for GIN layer
            mlp = nn.Sequential(
                nn.Linear(dims[i], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dims[i + 1])
            )
            
            # GIN convolution
            self.convs.append(GINConv(
                nn=mlp,
                eps=eps,
                train_eps=train_eps
            ))
            
            # Batch normalization
            self.batch_norms.append(nn.BatchNorm1d(dims[i + 1]))

    def forward(self, x, edge_index, batch=None):
        """Forward pass through GIN layers."""
        # Node-level embeddings
        h = x
        
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.batch_norms[i](h)
            
            if i < self.num_layers - 1:  # Don't apply activation/dropout to final layer
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # For node embeddings, return the node features directly
        if batch is None:
            return h
        
        # For graph-level tasks, apply pooling
        return self.pool(h, batch)

    def encode(self, x, edge_index, batch=None):
        """Encode nodes/graphs to embeddings (alias for forward)."""
        return self.forward(x, edge_index, batch)


@register_embedding("gin", aliases=["graph_isomorphism_network"])
class GINEmbedding(InductiveGraphEmbedding):
    """
    Graph Isomorphism Network (GIN) embedding method.

    GIN is designed to be as powerful as the Weisfeiler-Lehman test and is
    particularly effective for graph classification tasks. It uses learnable
    aggregation functions that can theoretically distinguish any graphs that
    the WL test can distinguish.
    
    This implementation focuses on node-level embeddings but can be extended
    for graph-level tasks by using appropriate pooling.

    Args:
        embedding_dim: Dimensionality of output embeddings
        hidden_dim: Hidden layer dimensionality (default: 256)
        num_layers: Number of GIN layers (default: 3)
        learning_rate: Learning rate for optimization (default: 0.001)
        epochs: Number of training epochs (default: 300)
        eps: Initial epsilon value for GIN (default: 0.0)
        train_eps: Whether to learn epsilon (default: True)
        dropout: Dropout rate (default: 0.1)
        pooling: Pooling method for graph-level tasks (default: "mean")
        weight_decay: Weight decay for regularization (default: 5e-4)
        device: Device to run on (default: "cpu")
        use_node_features: Whether to use node features (default: True)
        feature_dim: Dimension of node features if not using data.x (default: None)
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        learning_rate: float = 0.001,
        epochs: int = 300,
        eps: float = 0.0,
        train_eps: bool = True,
        dropout: float = 0.1,
        pooling: Literal["mean", "max", "add"] = "mean",
        weight_decay: float = 5e-4,
        device: str = "cpu",
        use_node_features: bool = True,
        feature_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(embedding_dim, **kwargs)
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.eps = eps
        self.train_eps = train_eps
        self.dropout = dropout
        self.pooling = pooling
        self.weight_decay = weight_decay
        self.device = torch.device(device)
        self.use_node_features = use_node_features
        self.feature_dim = feature_dim
        
        self.model = None
        self.optimizer = None

    def _prepare_data(self, graph):
        """Convert graph to PyTorch Geometric Data format."""
        # Convert to PyG Data if needed
        if not isinstance(graph, Data):
            data = convert_graph_format(graph, target_format="pyg")
        else:
            data = graph.clone()
        
        # Handle node features
        if data.x is None or not self.use_node_features:
            # Create identity features if no node features available
            input_dim = self.feature_dim if self.feature_dim else data.num_nodes
            if data.num_nodes <= input_dim:
                # Use one-hot encoding for small graphs
                data.x = torch.eye(data.num_nodes, dtype=torch.float)
            else:
                # Use random features for large graphs
                data.x = torch.randn(data.num_nodes, input_dim)
        
        return data.to(self.device)

    def fit(self, graph, verbose: bool = True):
        """
        Fit the GIN model to the graph.
        
        Args:
            graph: Input graph in supported format
            verbose: Whether to print training progress
            
        Returns:
            self: The fitted embedder
        """
        # Prepare data
        data = self._prepare_data(graph)
        input_dim = data.x.size(1)
        
        # Initialize model
        self.model = GINModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            eps=self.eps,
            train_eps=self.train_eps,
            pooling=self.pooling
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Training loop (self-supervised using edge prediction)
        self.model.train()
        
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            
            # Get embeddings
            embeddings = self.model(data.x, data.edge_index)
            
            # Self-supervised loss using edge prediction
            # Positive edges
            pos_edge_index = data.edge_index
            pos_scores = (embeddings[pos_edge_index[0]] * embeddings[pos_edge_index[1]]).sum(dim=1)
            
            # Negative edges
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=pos_edge_index.size(1)
            )
            neg_scores = (embeddings[neg_edge_index[0]] * embeddings[neg_edge_index[1]]).sum(dim=1)
            
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
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1:3d}/{self.epochs}: Loss = {loss.item():.4f}")
        
        if verbose:
            print(f"GIN training completed after {self.epochs} epochs")
            
        return self

    def transform(self, graph):
        """
        Generate embeddings for the graph.
        
        Args:
            graph: Input graph in supported format
            
        Returns:
            torch.Tensor: Node embeddings of shape (num_nodes, embedding_dim)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare data
        data = self._prepare_data(graph)
        
        # Generate embeddings
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(data.x, data.edge_index)
        
        return embeddings.cpu()

    def fit_transform(self, graph, verbose: bool = True):
        """
        Fit the model and generate embeddings.
        
        Args:
            graph: Input graph in supported format
            verbose: Whether to print training progress
            
        Returns:
            torch.Tensor: Node embeddings
        """
        return self.fit(graph, verbose=verbose).transform(graph)

    def transform_new_nodes(self, graph, node_indices=None):
        """
        Generate embeddings for new nodes (inductive capability).
        
        Args:
            graph: Input graph containing new nodes
            node_indices: Specific node indices to embed (optional)
            
        Returns:
            torch.Tensor: Embeddings for new nodes
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare data
        data = self._prepare_data(graph)
        
        # Generate embeddings
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(data.x, data.edge_index)
        
        # Return specific node embeddings if requested
        if node_indices is not None:
            return embeddings[node_indices].cpu()
        
        return embeddings.cpu()