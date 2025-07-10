"""
Training utilities for graph embedding methods.

Contains specialized training functions for different embedding approaches.
"""

import torch
from .utils import EarlyStopping


def train_gae(model, data, epochs=1000, lr=0.01, weight_decay=0.0, variational=False, patience=None, verbose=False):
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
        Trained model
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Initialize early stopping if patience is provided
    early_stopping = EarlyStopping(patience=patience) if patience is not None else None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        z = model.encode(data.x, data.edge_index)
        
        # Compute loss
        if variational:
            # VGAE loss includes KL divergence
            loss = model.recon_loss(z, data.edge_index)
            if hasattr(model, 'kl_loss'):
                loss += (1 / data.num_nodes) * model.kl_loss()
        else:
            # GAE loss is just reconstruction loss
            loss = model.recon_loss(z, data.edge_index)
        
        loss.backward()
        optimizer.step()
        
        # Print progress
        if verbose and (epoch % 50 == 0 or epoch < 10):
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
        
        # Check early stopping
        if early_stopping is not None:
            if early_stopping(loss.item(), model):
                if verbose:
                    print(f'Early stopping at epoch {epoch}, Loss: {loss:.4f}')
                break
    
    return model
