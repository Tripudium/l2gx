"""
GraphSAGE embedding method.

This module provides GraphSAGE (Graph Sample and Aggregate) embedding implementation
that follows the standard GraphEmbedding interface. GraphSAGE is an inductive method
that learns embeddings by sampling and aggregating features from node neighborhoods.
"""

from typing import Union, List, Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling
import networkx as nx
import scipy.sparse as sp

from ..base import InductiveGraphEmbedding
from ..registry import register_embedding
from ..utils import convert_graph_format


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE model implementation using PyTorch Geometric.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout: float = 0.5, 
                 aggregator: Literal['mean', 'max', 'lstm'] = 'mean'):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create SAGE layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=aggregator))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, output_dim, aggr=aggregator))
        else:
            # Single layer case
            self.convs[0] = SAGEConv(input_dim, output_dim, aggr=aggregator)
    
    def forward(self, x, edge_index):
        """Forward pass through GraphSAGE layers."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Don't apply activation/dropout to final layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def encode(self, x, edge_index):
        """Encode nodes to embeddings (alias for forward)."""
        return self.forward(x, edge_index)


@register_embedding('graphsage', aliases=['sage', 'graph_sage'])
class GraphSAGEEmbedding(InductiveGraphEmbedding):
    """
    GraphSAGE (Graph Sample and Aggregate) embedding method.
    
    GraphSAGE learns node embeddings by sampling and aggregating features from
    a node's local neighborhood. It's an inductive method that can generate
    embeddings for previously unseen nodes.
    """
    
    def __init__(self, 
                 embedding_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 learning_rate: float = 0.01,
                 epochs: int = 200,
                 batch_size: int = 256,
                 num_neighbors: List[int] = None,
                 aggregator: Literal['mean', 'max', 'lstm'] = 'mean',
                 dropout: float = 0.5,
                 weight_decay: float = 5e-4,
                 device: str = 'cpu',
                 **kwargs):
        """
        Initialize GraphSAGE embedding method.
        
        Args:
            embedding_dim: Output embedding dimensionality
            hidden_dim: Hidden layer dimensionality
            num_layers: Number of GraphSAGE layers
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            batch_size: Batch size for training
            num_neighbors: Number of neighbors to sample at each layer
            aggregator: Aggregation function ('mean', 'max', 'lstm')
            dropout: Dropout probability
            weight_decay: Weight decay for regularization
            device: Device to run computations on ('cpu' or 'cuda')
            **kwargs: Additional parameters
        """
        super().__init__(embedding_dim, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors or [10, 5]  # Default neighbor sampling
        self.aggregator = aggregator
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.device = device
        
        self._model = None
        self._data = None
        self._node_features = None
        
    def fit(self, graph: Union[Data, nx.Graph, sp.spmatrix, np.ndarray]) -> 'GraphSAGEEmbedding':
        """
        Fit the GraphSAGE model to the graph.
        
        Args:
            graph: Input graph in supported format
            
        Returns:
            self: Fitted embedding instance
        """
        self._validate_graph_format(graph)
        
        # Convert to PyTorch Geometric format
        self._data = convert_graph_format(graph, target_format='torch_geometric')
        
        # Handle node features
        if self._data.x is None:
            # Use identity features if no features provided
            self._data.x = torch.eye(self._data.num_nodes, dtype=torch.float)
        
        self._node_features = self._data.x
        input_dim = self._data.x.size(1)
        
        # Initialize model
        self._model = GraphSAGEModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            aggregator=self.aggregator
        ).to(self.device)
        
        # Move data to device
        self._data = self._data.to(self.device)
        
        # Train the model
        self._train_unsupervised()
        
        self.is_fitted = True
        return self
    
    def _train_unsupervised(self):
        """Train GraphSAGE using unsupervised learning with negative sampling."""
        optimizer = torch.optim.Adam(
            self._model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self._model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = self._model(self._data.x, self._data.edge_index)
            
            # Negative sampling for unsupervised loss
            neg_edge_index = negative_sampling(
                edge_index=self._data.edge_index,
                num_nodes=self._data.num_nodes,
                num_neg_samples=self._data.edge_index.size(1)
            )
            
            # Compute unsupervised loss (link prediction)
            pos_score = self._link_prediction_score(embeddings, self._data.edge_index)
            neg_score = self._link_prediction_score(embeddings, neg_edge_index)
            
            # Binary cross-entropy loss
            pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
            loss = pos_loss + neg_loss
            
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    
    def _link_prediction_score(self, embeddings, edge_index):
        """Compute link prediction scores for given edges."""
        row, col = edge_index
        return (embeddings[row] * embeddings[col]).sum(dim=1)
    
    def transform(self, graph: Union[Data, nx.Graph, sp.spmatrix, np.ndarray]) -> np.ndarray:
        """
        Generate node embeddings using the fitted GraphSAGE model.
        
        Args:
            graph: Input graph (can include new nodes for inductive setting)
            
        Returns:
            Node embeddings of shape (num_nodes, embedding_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        # Convert input graph to PyTorch Geometric format
        data = convert_graph_format(graph, target_format='torch_geometric')
        
        # Handle node features for new graph
        if data.x is None:
            # Use identity features
            data.x = torch.eye(data.num_nodes, dtype=torch.float)
        
        # Move to device
        data = data.to(self.device)
        
        self._model.eval()
        with torch.no_grad():
            embeddings = self._model(data.x, data.edge_index)
            
        return embeddings.cpu().numpy()
    
    def transform_new_nodes(self, 
                           graph: Union[Data, nx.Graph, sp.spmatrix, np.ndarray],
                           new_node_indices: np.ndarray) -> np.ndarray:
        """
        Generate embeddings for new nodes without retraining.
        
        This is the key advantage of GraphSAGE - it can embed new nodes
        by aggregating from their neighborhoods.
        
        Args:
            graph: Input graph containing both old and new nodes
            new_node_indices: Indices of the new nodes to embed
            
        Returns:
            Embeddings for the new nodes of shape (len(new_node_indices), embedding_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform_new_nodes()")
        
        # Get embeddings for the full graph
        all_embeddings = self.transform(graph)
        
        # Return embeddings for new nodes only
        return all_embeddings[new_node_indices]
    
    def get_model(self) -> GraphSAGEModel:
        """
        Get the underlying GraphSAGE model.
        
        Returns:
            Trained GraphSAGE model instance
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before getting model")
        return self._model