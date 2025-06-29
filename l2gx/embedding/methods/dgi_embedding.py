"""
Deep Graph Infomax (DGI) embedding method.

This module provides DGI embedding implementation that follows the standard
GraphEmbedding interface. DGI learns node embeddings by maximizing mutual
information between patch representations and global graph summaries.
"""

from typing import Union, Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import networkx as nx
import scipy.sparse as sp

from ..base import InductiveGraphEmbedding
from ..registry import register_embedding
from ..utils import convert_graph_format


class Encoder(nn.Module):
    """
    Graph encoder for DGI.
    
    Supports different encoder architectures (GCN, GAT, GraphSAGE).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, 
                 encoder_type: Literal['gcn', 'gat', 'sage'] = 'gcn',
                 dropout: float = 0.0):
        super().__init__()
        
        self.encoder_type = encoder_type
        self.dropout = dropout
        
        if encoder_type == 'gcn':
            self.conv = GCNConv(input_dim, hidden_dim)
        elif encoder_type == 'gat':
            self.conv = GATConv(input_dim, hidden_dim, dropout=dropout)
        elif encoder_type == 'sage':
            self.conv = SAGEConv(input_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    def forward(self, x, edge_index):
        """Forward pass through encoder."""
        h = self.conv(x, edge_index)
        if self.encoder_type != 'gat':  # GAT already applies activation internally
            h = F.relu(h)
        return h


class Discriminator(nn.Module):
    """
    Discriminator for DGI.
    
    Distinguishes between positive and negative patch-summary pairs.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.weight = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize discriminator parameters."""
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, h, summary, nodes=None):
        """
        Compute discriminator scores.
        
        Args:
            h: Node embeddings [num_nodes, hidden_dim]
            summary: Graph summary [hidden_dim]
            nodes: Node indices to score (if None, score all nodes)
            
        Returns:
            Discriminator scores
        """
        if nodes is not None:
            h = h[nodes]
        
        # Compute compatibility scores
        summary = torch.matmul(summary, self.weight)
        scores = torch.sum(h * summary, dim=1)
        
        return scores


class Readout(nn.Module):
    """
    Readout function to create graph-level summary.
    
    Supports different aggregation methods.
    """
    
    def __init__(self, readout_type: Literal['mean', 'max', 'sum'] = 'mean'):
        super().__init__()
        self.readout_type = readout_type
    
    def forward(self, h):
        """
        Create graph summary from node embeddings.
        
        Args:
            h: Node embeddings [num_nodes, hidden_dim]
            
        Returns:
            Graph summary [hidden_dim]
        """
        if self.readout_type == 'mean':
            return torch.mean(h, dim=0)
        elif self.readout_type == 'max':
            return torch.max(h, dim=0)[0]
        elif self.readout_type == 'sum':
            return torch.sum(h, dim=0)
        else:
            raise ValueError(f"Unknown readout type: {self.readout_type}")


class DGIModel(nn.Module):
    """
    Complete DGI model implementation.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int,
                 encoder_type: Literal['gcn', 'gat', 'sage'] = 'gcn',
                 readout_type: Literal['mean', 'max', 'sum'] = 'mean',
                 dropout: float = 0.0):
        super().__init__()
        
        self.encoder = Encoder(input_dim, hidden_dim, encoder_type, dropout)
        self.discriminator = Discriminator(hidden_dim)
        self.readout = Readout(readout_type)
        
        self.hidden_dim = hidden_dim
    
    def forward(self, x, edge_index):
        """Forward pass to get node embeddings."""
        return self.encoder(x, edge_index)
    
    def compute_loss(self, x, edge_index, corrupt_x=None):
        """
        Compute DGI loss using mutual information maximization.
        
        Args:
            x: Node features
            edge_index: Graph edges
            corrupt_x: Corrupted node features (if None, will be generated)
            
        Returns:
            DGI loss
        """
        # Positive samples
        pos_h = self.encoder(x, edge_index)
        pos_summary = self.readout(pos_h)
        
        # Negative samples (corrupted features)
        if corrupt_x is None:
            corrupt_x = self._corrupt_features(x)
        
        neg_h = self.encoder(corrupt_x, edge_index)
        neg_summary = self.readout(neg_h)
        
        # Discriminator scores
        pos_scores = self.discriminator(pos_h, pos_summary)
        neg_scores = self.discriminator(neg_h, neg_summary)
        
        # Binary cross-entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )
        
        return pos_loss + neg_loss
    
    def _corrupt_features(self, x):
        """Corrupt node features by row-wise shuffling."""
        corrupt_x = x.clone()
        
        # Shuffle each feature dimension independently
        for i in range(x.size(1)):
            perm = torch.randperm(x.size(0))
            corrupt_x[:, i] = x[perm, i]
        
        return corrupt_x


@register_embedding('dgi', aliases=['deep_graph_infomax', 'infomax'])
class DGIEmbedding(InductiveGraphEmbedding):
    """
    Deep Graph Infomax (DGI) embedding method.
    
    DGI learns node embeddings by maximizing mutual information between 
    patch-level representations and graph-level summaries using a discriminator.
    It's particularly effective for self-supervised learning on graphs.
    """
    
    def __init__(self, 
                 embedding_dim: int,
                 encoder_type: Literal['gcn', 'gat', 'sage'] = 'gcn',
                 readout_type: Literal['mean', 'max', 'sum'] = 'mean',
                 learning_rate: float = 0.001,
                 epochs: int = 300,
                 dropout: float = 0.0,
                 weight_decay: float = 0.0,
                 patience: int = 20,
                 device: str = 'cpu',
                 **kwargs):
        """
        Initialize DGI embedding method.
        
        Args:
            embedding_dim: Output embedding dimensionality  
            encoder_type: Graph encoder architecture ('gcn', 'gat', 'sage')
            readout_type: Graph summary aggregation ('mean', 'max', 'sum')
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            dropout: Dropout probability
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
            device: Device to run computations on ('cpu' or 'cuda')
            **kwargs: Additional parameters
        """
        super().__init__(embedding_dim, **kwargs)
        self.encoder_type = encoder_type
        self.readout_type = readout_type
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device
        
        self._model = None
        self._data = None
        
    def fit(self, graph: Union[Data, nx.Graph, sp.spmatrix, np.ndarray]) -> 'DGIEmbedding':
        """
        Fit the DGI model to the graph.
        
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
        
        input_dim = self._data.x.size(1)
        
        # Initialize model
        self._model = DGIModel(
            input_dim=input_dim,
            hidden_dim=self.embedding_dim,
            encoder_type=self.encoder_type,
            readout_type=self.readout_type,
            dropout=self.dropout
        ).to(self.device)
        
        # Move data to device
        self._data = self._data.to(self.device)
        
        # Train the model
        self._train()
        
        self.is_fitted = True
        return self
    
    def _train(self):
        """Train DGI using mutual information maximization."""
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        best_loss = float('inf')
        patience_counter = 0
        
        self._model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Compute DGI loss
            loss = self._model.compute_loss(self._data.x, self._data.edge_index)
            
            loss.backward()
            optimizer.step()
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f'Early stopping at epoch {epoch}')
                break
            
            if epoch % 50 == 0:
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    
    def transform(self, graph: Union[Data, nx.Graph, sp.spmatrix, np.ndarray]) -> np.ndarray:
        """
        Generate node embeddings using the fitted DGI model.
        
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
        
        DGI can handle new nodes inductively by encoding their features
        and local graph structure.
        
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
    
    def get_model(self) -> DGIModel:
        """
        Get the underlying DGI model.
        
        Returns:
            Trained DGI model instance
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before getting model")
        return self._model
    
    def get_graph_summary(self, graph: Union[Data, nx.Graph, sp.spmatrix, np.ndarray]) -> np.ndarray:
        """
        Get graph-level summary representation.
        
        Args:
            graph: Input graph
            
        Returns:
            Graph summary vector of shape (embedding_dim,)
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before getting graph summary")
        
        # Get node embeddings
        node_embeddings = self.transform(graph)
        node_embeddings_tensor = torch.tensor(node_embeddings, dtype=torch.float).to(self.device)
        
        # Apply readout function
        self._model.eval()
        with torch.no_grad():
            summary = self._model.readout(node_embeddings_tensor)
            
        return summary.cpu().numpy()