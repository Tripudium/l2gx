"""
Model for aligning patch embeddings
"""

import torch
from torch import nn
import geotorch


# pylint: disable=too-few-public-methods
class OrthogonalModel(nn.Module):
    """
    Model for aligning patch embeddings
    """

    def __init__(self, dim, n_patches, device):
        """
        Initialize the model
        Args:
            dim: int
            n_patches: int
            device: str
        """
        super().__init__()
        self.device = device
        linear_layers = [
            nn.Linear(dim, dim, bias=True).to(device) for _ in range(n_patches)
        ]
        fixed_layer_index = 0
        linear_layers[fixed_layer_index].weight.data.copy_(torch.eye(dim))
        linear_layers[fixed_layer_index].weight.requires_grad = False
        self.transformation = nn.ParameterList(linear_layers)
        for i in range(1, n_patches):
            geotorch.orthogonal(self.transformation[i], "weight")
            self.transformation[i].weight = torch.eye(dim)
        

    def forward(self, patch_intersection):
        """
        Forward pass
        Args:
            patch_intersection: list of tuples
        Returns:
            list of transformed embeddings
        """
        outputs = {}
        for (i, j), (X, Y) in patch_intersection.items():
            Xt = self.transformation[i](X)
            Yt = self.transformation[j](Y)
            outputs[(i, j)] = (Xt, Yt)
        return outputs


# pylint: disable=too-few-public-methods
class AffineModel(nn.Module):
    """
    Model for aligning patch embeddings
    """

    def __init__(self, dim, n_patches, device):
        """
        Initialize the model
        Args:
            dim: int
            n_patches: int
            device: str
        """
        super().__init__()
        self.device = device
        linear_layers = [
            nn.Linear(dim, dim, bias=True).to(device) for _ in range(n_patches)
        ]
        # Fix the first transformation to be the identity
        fixed_layer_index = 0
        linear_layers[fixed_layer_index].bias.data.zero_()
        linear_layers[fixed_layer_index].weight.data.copy_(torch.eye(dim))
        linear_layers[fixed_layer_index].weight.requires_grad = False
        linear_layers[fixed_layer_index].bias.requires_grad = False

        self.transformation = nn.ParameterList(linear_layers)

    def forward(self, patch_intersection):
        """
        Forward pass
        """
        outputs = {}
        for (i, j), (X, Y) in patch_intersection.items():
            Xt = self.transformation[i](X)
            Yt = self.transformation[j](Y)
            outputs[(i, j)] = (Xt, Yt)
        return outputs
