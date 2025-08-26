"""
Utilities for graph embedding operations.

Provides graph format conversion and utility functions for embedding methods.
"""

import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
import networkx as nx
import scipy.sparse as sp
from typing import Union, Literal


import tempfile
import torch


class EarlyStopping:
    """Early stopping utility for training with model state management."""

    def __init__(self, patience=10, min_delta=0, save_best_model=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.save_best_model = save_best_model
        self._temp_file = None

        if self.save_best_model:
            self._temp_file = tempfile.NamedTemporaryFile(delete=False)

    def __call__(self, val_loss, model=None):
        improved = False

        if self.best_loss is None:
            self.best_loss = val_loss
            improved = True
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            improved = True
        else:
            self.counter += 1

        # Save best model state if improved
        if improved and self.save_best_model and model is not None:
            torch.save(model.state_dict(), self._temp_file.name)

        # Check if we should stop
        should_stop = self.counter >= self.patience

        # Restore best model if stopping
        if should_stop and self.save_best_model and model is not None:
            model.load_state_dict(torch.load(self._temp_file.name))

        return should_stop

    def __del__(self):
        # Clean up temporary file
        if self._temp_file is not None:
            try:
                import os

                os.unlink(self._temp_file.name)
            except:
                pass


def convert_graph_format(
    graph: Union[Data, nx.Graph, sp.spmatrix, np.ndarray],
    target_format: Literal["torch_geometric", "networkx", "scipy_sparse", "numpy"],
) -> Union[Data, nx.Graph, sp.spmatrix, np.ndarray]:
    """
    Convert graph between different formats.

    Args:
        graph: Input graph in any supported format
        target_format: Target format to convert to

    Returns:
        Graph in the target format
    """
    # First, convert to a common intermediate format (scipy sparse)
    if isinstance(graph, Data):
        adj_matrix = to_scipy_sparse_matrix(graph.edge_index, num_nodes=graph.num_nodes)
        node_features = graph.x
    elif isinstance(graph, nx.Graph):
        adj_matrix = nx.adjacency_matrix(graph)
        node_features = None
    elif isinstance(graph, sp.spmatrix):
        adj_matrix = graph
        node_features = None
    elif isinstance(graph, np.ndarray):
        adj_matrix = sp.csr_matrix(graph)
        node_features = None
    else:
        raise TypeError(f"Unsupported input graph type: {type(graph)}")

    # Now convert to target format
    if target_format == "scipy_sparse":
        return adj_matrix
    if target_format == "numpy":
        return adj_matrix.toarray()
    if target_format == "networkx":
        return nx.from_scipy_sparse_array(adj_matrix)
    if target_format == "torch_geometric":
        edge_index, edge_attr = from_scipy_sparse_matrix(adj_matrix)
        data = Data(
            edge_index=edge_index, edge_attr=edge_attr, num_nodes=adj_matrix.shape[0]
        )
        if node_features is not None:
            data.x = node_features
        return data

    raise ValueError(f"Unsupported target format: {target_format}")
