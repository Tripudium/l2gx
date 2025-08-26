"""
Training utilities for graph embedding methods.

Contains specialized training functions for different embedding approaches.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .utils import EarlyStopping


def train_gae(
    model: nn.Module,
    data: Data,
    epochs: int = 1000,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    variational=False,
    patience: int | None = None,
    verbose: bool = False,
):
    """
    Train a GAE or VGAE model.

    Args:
        model: GAE or VGAE model instance
        data: PyTorch Geometric Data object
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer (default: 0.0)
        variational: Whether this is a variational model (VGAE)
        patience: Early stopping patience (None to disable early stopping)
        verbose: Whether to print training progress

    Returns:
        tuple of (trained model, training history dict)
        Training history dict contains:
            - 'epochs': list of epoch numbers
            - 'losses': list of loss values per epoch
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize early stopping if patience is provided
    early_stopping = EarlyStopping(patience=patience) if patience is not None else None

    # Initialize training history
    training_history = {
        'epochs': [],
        'losses': []
    }

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        z = model.encode(data.x, data.edge_index)

        # Compute loss
        if variational:
            # VGAE loss includes KL divergence
            loss = model.recon_loss(z, data.edge_index)
            if hasattr(model, "kl_loss"):
                loss += (1 / data.num_nodes) * model.kl_loss()
        else:
            # GAE loss is just reconstruction loss
            loss = model.recon_loss(z, data.edge_index)

        loss.backward()
        optimizer.step()

        # Record training history
        training_history['epochs'].append(epoch)
        training_history['losses'].append(loss.item())

        # Print progress
        if verbose and (epoch % 50 == 0 or epoch < 10):
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")

        # Check early stopping
        if early_stopping is not None and early_stopping(loss.item(), model):
            if verbose:
                print(f"Early stopping at epoch {epoch}, Loss: {loss:.4f}")
            break

    return model, training_history
