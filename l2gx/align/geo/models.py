"""
Geometric alignment models.
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal


class GeoModel(nn.Module):
    """
    Geometric alignment model with one affine transformation per patch.

    The model applies patch-specific affine transformations to input vectors.
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

        # Create one affine transformation for each patch
        self.transformations = nn.ModuleList(
            [nn.Linear(dim, dim, bias=use_bias) for _ in range(n_patches)]
        )

        # Initialize all transformations closer to identity to reduce skewing
        with torch.no_grad():
            for i, transformation in enumerate(self.transformations):
                # Initialize weights as identity + small random perturbation
                W = torch.eye(dim) + 0.01 * torch.randn(dim, dim)

                # Ensure positive determinant to avoid reflections
                # (not sure if this is necessary)
                if torch.det(W) < 0:
                    W[:, 0] *= -1

                transformation.weight.copy_(W)
                if use_bias:
                    transformation.bias.zero_()

        self.to(device)

    def forward(
        self, i: int, j: int, x_i: torch.Tensor, x_j: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for pre-generated batches.

        Processes pre-generated batches where:
        - x_i contains coordinates from patch i for nodes in intersection of patches i and j
        - x_j contains coordinates from patch j for nodes in intersection of patches i and j

        Args:
            i: First patch index
            j: Second patch index
            x_i: Coordinates from patch i for intersection nodes, shape (batch_size, dim)
            x_j: Coordinates from patch j for intersection nodes, shape (batch_size, dim)

        Returns:
            tuple of (y_i, y_j) transformed coordinates
        """
        y_i = self.transformations[i](x_i)
        y_j = self.transformations[j](x_j)
        return y_i, y_j


class GeoModelOrthogonal(nn.Module):
    """
    Geometric alignment model with orthogonal parametrization.

    The model applies patch-specific orthogonal transformations to input vectors.
    """

    def __init__(
        self,
        device: str,
        n_patches: int,
        dim: int,
        use_bias: bool = True,
        initial_rotations=None,
    ):
        """
        Initialize the GeoModelOrthogonal.

        Args:
            device: Device to place the model on ('cpu' or 'cuda')
            n_patches: Number of patches
            dim: Dimension of the input/output vectors
            use_bias: Whether to use bias terms (should be False when patches are centered)
            initial_rotations: list of initial rotation matrices (optional)
        """
        super().__init__()
        self.device = device
        self.n_patches = n_patches
        self.dim = dim
        self.use_bias = use_bias

        # Create one affine transformation for each patch
        self.transformations = nn.ModuleList()

        for i in range(n_patches):
            layer = nn.Linear(dim, dim, bias=use_bias)

            # Initialize weights
            with torch.no_grad():
                if initial_rotations is not None and i < len(initial_rotations):
                    # Use precomputed rotation (already transposed in _set_model_initial_rotations)
                    initial_weight = torch.tensor(
                        initial_rotations[i].T, dtype=torch.float32
                    )
                    layer.weight.copy_(initial_weight)
                else:
                    # Initialize with identity + small random perturbation
                    W = torch.eye(dim) + 0.01 * torch.randn(dim, dim)

                    # Apply QR decomposition to get an orthogonal matrix close to identity
                    Q, _ = torch.linalg.qr(W)
                    # Ensure positive determinant (no reflections)
                    if torch.det(Q) < 0:
                        Q[:, 0] *= -1
                    layer.weight.copy_(Q)

                if use_bias:
                    layer.bias.zero_()

            # Apply orthogonal parametrization
            layer = orthogonal(layer, name="weight")

            self.transformations.append(layer)

        self.to(device)

    def forward(
        self, i: int, j: int, x_i: torch.Tensor, x_j: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for pre-generated batches.

        Processes pre-generated batches where:
        - x_i contains coordinates from patch i for nodes in intersection of patches i and j
        - x_j contains coordinates from patch j for nodes in intersection of patches i and j

        Args:
            i: First patch index
            j: Second patch index
            x_i: Coordinates from patch i for intersection nodes, shape (batch_size, dim)
            x_j: Coordinates from patch j for intersection nodes, shape (batch_size, dim)

        Returns:
            tuple of (y_i, y_j) transformed coordinates
        """
        y_i = self.transformations[i](x_i)
        y_j = self.transformations[j](x_j)
        return y_i, y_j
