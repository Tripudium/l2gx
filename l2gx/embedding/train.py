"""
Training utilities for graph embedding methods.

Contains specialized training functions for different embedding approaches.
"""

import torch


def train_gae(model, data, epochs=200, lr=0.01, variational=False):
    """
    Train a GAE or VGAE model.
    
    Args:
        model: GAE or VGAE model instance
        data: PyTorch Geometric Data object
        epochs: Number of training epochs
        lr: Learning rate
        variational: Whether this is a variational model (VGAE)
        
    Returns:
        Trained model
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
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
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    
    return model
