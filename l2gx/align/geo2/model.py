"""
GeoModel for the redesigned geometric aligner.
"""

import torch
import torch.nn as nn


class GeoModel(nn.Module):
    """
    Geometric alignment model with one linear layer per patch.
    
    The model applies patch-specific linear transformations to input vectors.
    """
    
    def __init__(self, device: str, n_patches: int, dim: int, use_bias: bool = True):
        """
        Initialize the GeoModel.
        
        Args:
            device: Device to place the model on ('cpu' or 'cuda')
            n_patches: Number of patches
            dim: Dimension of the input/output vectors
            use_bias: Whether to use bias terms (should be False when patches are centered)
        """
        super().__init__()
        self.device = device
        self.n_patches = n_patches
        self.dim = dim
        self.use_bias = use_bias
        
        # Create one linear layer for each patch
        self.transformations = nn.ModuleList([
            nn.Linear(dim, dim, bias=use_bias) for _ in range(n_patches)
        ])
        
        # Fix the first transformation to identity (reference patch)
        with torch.no_grad():
            self.transformations[0].weight.copy_(torch.eye(dim))
            if use_bias:
                self.transformations[0].bias.zero_()
            self.transformations[0].weight.requires_grad = False
            if use_bias:
                self.transformations[0].bias.requires_grad = False
        
        # Move to device
        self.to(device)
    
    def forward(self, patch_indices: tuple[int, int], vectors: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying transformations to input vectors.
        
        Args:
            patch_indices: Tuple (i, j) of patch indices
            vectors: Tensor of shape (2, dim) for single example or (2, batch_size, dim) for batch
        
        Returns:
            Tensor of same shape as input with transformations applied
        """
        i, j = patch_indices
        
        if vectors.dim() == 2:
            # Single example: shape (2, dim)
            x_1, x_2 = vectors[0], vectors[1]
            y_1 = self.transformations[i](x_1)
            y_2 = self.transformations[j](x_2)
            return torch.stack([y_1, y_2], dim=0)
        else:
            # Batch: shape (2, batch_size, dim)
            x_1, x_2 = vectors[0], vectors[1]  # Each is (batch_size, dim)
            y_1 = self.transformations[i](x_1)
            y_2 = self.transformations[j](x_2)
            return torch.stack([y_1, y_2], dim=0)  # (2, batch_size, dim)
    
    def forward_batch(self, i: int, j: int, x_i: torch.Tensor, x_j: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized forward pass for batched data.
        
        Args:
            i: First patch index
            j: Second patch index  
            x_i: Coordinates from patch i, shape (batch_size, dim)
            x_j: Coordinates from patch j, shape (batch_size, dim)
            
        Returns:
            Tuple of (y_i, y_j) transformed coordinates
        """
        y_i = self.transformations[i](x_i)
        y_j = self.transformations[j](x_j)
        return y_i, y_j