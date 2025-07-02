"""
Redesigned geometric alignment implementation.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional

from l2gx.align.registry import register_aligner
from l2gx.align.alignment import AlignmentProblem
from l2gx.patch import Patch
from l2gx.graphs.tgraph import TGraph
from .model import GeoModel


@register_aligner("geo2")
class GeoAlignmentProblem2(AlignmentProblem):
    """
    Redesigned geometric alignment using patch graphs and simplified model.
    """

    def __init__(self, verbose: bool = False, min_overlap: Optional[int] = None,
                 use_orthogonal_reg: bool = True, orthogonal_reg_weight: float = 10.0,
                 batch_size: int = 512, center_patches: bool = True):
        """
        Initialize the geometric alignment problem.
        
        Args:
            verbose: Whether to print debug information
            min_overlap: Minimum overlap required between patches
            use_orthogonal_reg: Whether to use orthogonal regularization
            orthogonal_reg_weight: Weight for orthogonal regularization term
            batch_size: Batch size for training (0 = full batch)
            center_patches: Whether to center patches before applying transformations
        """
        super().__init__(verbose=verbose, min_overlap=min_overlap)
        self.patch_graph = None
        self.model = None
        self.training_data = []
        self.loss_history = []
        self.use_orthogonal_reg = use_orthogonal_reg
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.batch_size = batch_size
        self.center_patches = center_patches
        self.patch_centers = {}

    def _create_patch_graph(self, patches: list[Patch], min_overlap: int) -> TGraph:
        """
        Create a patch graph with nodes for patches and edges for sufficient overlaps.
        
        Args:
            patches: List of patches
            min_overlap: Minimum overlap threshold
            
        Returns:
            TGraph representing the patch connectivity
            
        Raises:
            RuntimeError: If the patch graph is not connected
        """
        if self.verbose:
            print(f"Creating patch graph with min_overlap={min_overlap}")
        
        # Find overlaps between patches and build edge list
        edges = []
        overlaps = {}
        
        for i in range(len(patches)):
            for j in range(i + 1, len(patches)):
                # Find overlapping nodes
                overlap_nodes = list(set(patches[i].nodes) & set(patches[j].nodes))
                
                if len(overlap_nodes) >= min_overlap:
                    edges.append((i, j))
                    overlaps[(i, j)] = overlap_nodes
                    
                    if self.verbose:
                        print(f"  Edge ({i}, {j}): {len(overlap_nodes)} overlapping nodes")
        
        # Check connectivity using simple BFS
        if len(edges) == 0:
            if len(patches) > 1:
                raise RuntimeError(
                    "Patch graph is not connected. Found 0 edges between patches. "
                    "Consider reducing min_overlap or increasing patch overlaps."
                )
        else:
            # Build adjacency list for connectivity check
            adj_list = {i: [] for i in range(len(patches))}
            for i, j in edges:
                adj_list[i].append(j)
                adj_list[j].append(i)
            
            # BFS to check connectivity
            visited = set()
            queue = [0]
            visited.add(0)
            
            while queue:
                current_node = queue.pop(0)
                for neighbor in adj_list[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            if len(visited) != len(patches):
                num_components = len(patches) - len(visited) + 1
                raise RuntimeError(
                    f"Patch graph is not connected. Found {num_components} "
                    f"connected components. Consider reducing min_overlap or increasing patch overlaps."
                )
        
        if self.verbose:
            print(f"Patch graph created: {len(patches)} nodes, {len(edges)} edges")
        
        # Create edge_index tensor directly
        if len(edges) == 0:
            # No edges - create empty edge_index
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            # Convert edges to tensor and make undirected
            edge_tensor = torch.tensor(edges, dtype=torch.long).T
            reverse_edges = torch.stack([edge_tensor[1], edge_tensor[0]], dim=0)
            edge_index = torch.cat([edge_tensor, reverse_edges], dim=1)
        
        tgraph = TGraph(
            edge_index=edge_index,
            num_nodes=len(patches),
            undir=True
        )
        
        # Store overlaps for training data generation
        self.overlaps = overlaps
        
        return tgraph

    def _center_patches(self, patches: list[Patch]) -> list[Patch]:
        """
        Center patches by subtracting their mean coordinates.
        
        Args:
            patches: List of patches to center
            
        Returns:
            List of centered patches
        """
        if not self.center_patches:
            return patches
            
        centered_patches = []
        self.patch_centers = {}
        
        for i, patch in enumerate(patches):
            # Compute center
            center = patch.coordinates.mean(axis=0)
            self.patch_centers[i] = center
            
            # Create centered patch
            centered_coords = patch.coordinates - center
            centered_patch = Patch(patch.nodes.copy(), centered_coords)
            centered_patches.append(centered_patch)
            
            if self.verbose:
                print(f"  Patch {i}: centered at {center}")
        
        return centered_patches

    def _generate_training_data(self, patches: list[Patch], device: str = "cpu") -> dict:
        """
        Generate training data from patch overlaps in batched format.
        
        Args:
            patches: List of patches
            device: Device to place tensors on
            
        Returns:
            Dictionary with batched training data by patch pair
        """
        training_data = {}
        total_examples = 0
        
        for (i, j), overlap_nodes in self.overlaps.items():
            # Get coordinates for overlapping nodes in both patches
            coords_i = patches[i].get_coordinates(overlap_nodes)
            coords_j = patches[j].get_coordinates(overlap_nodes)
            
            # Create batched tensors for this patch pair
            x_i = torch.tensor(coords_i, dtype=torch.float32, device=device)  # Shape: (n_overlap, dim)
            x_j = torch.tensor(coords_j, dtype=torch.float32, device=device)  # Shape: (n_overlap, dim)
            
            training_data[(i, j)] = (x_i, x_j)
            total_examples += len(overlap_nodes)
        
        if self.verbose:
            print(f"Generated {total_examples} training examples in {len(training_data)} batches")
        
        return training_data

    def _compute_orthogonal_regularization(self, model: GeoModel) -> torch.Tensor:
        """
        Compute orthogonal regularization loss: ||W @ W.T - I||²_F
        
        Args:
            model: The GeoModel
            
        Returns:
            Orthogonal regularization loss (scalar tensor)
        """
        reg_loss = 0.0
        
        for i, transformation in enumerate(model.transformations):
            # Skip the fixed first transformation (identity)
            if i == 0:
                continue
                
            W = transformation.weight  # Shape: (dim, dim)
            
            # Compute W @ W.T
            WWT = torch.mm(W, W.T)
            
            # Compute ||W @ W.T - I||²_F
            identity = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
            diff = WWT - identity
            reg_loss += torch.sum(diff * diff)  # Frobenius norm squared
        
        return reg_loss

    def _compute_loss(self, model: GeoModel, training_data: dict) -> torch.Tensor:
        """
        Compute the total loss over all training examples using batched operations.
        
        Args:
            model: The GeoModel
            training_data: Dictionary of batched training data by patch pair
            
        Returns:
            Total loss (scalar tensor)
        """
        # Compute MSE loss using batched operations
        mse_loss = 0.0
        
        for (i, j), (x_i, x_j) in training_data.items():
            # Batched forward pass
            y_i, y_j = model.forward_batch(i, j, x_i, x_j)
            
            # Batched squared difference loss
            loss = F.mse_loss(y_i, y_j, reduction='sum')
            mse_loss += loss
        
        # Add orthogonal regularization if enabled
        total_loss = mse_loss
        if self.use_orthogonal_reg:
            reg_loss = self._compute_orthogonal_regularization(model)
            total_loss += self.orthogonal_reg_weight * reg_loss
        
        return total_loss

    def align_patches(
        self, 
        patches: list[Patch], 
        min_overlap: Optional[int] = None,
        num_epochs: int = 1000,
        learning_rate: float = 0.01,
        device: str = "cpu",
        use_orthogonal_reg: Optional[bool] = None,
        orthogonal_reg_weight: Optional[float] = None,
        batch_size: Optional[int] = None,
        center_patches: Optional[bool] = None
    ) -> 'GeoAlignmentProblem2':
        """
        Align patches using the redesigned geometric approach.
        
        Args:
            patches: List of patches to align
            min_overlap: Minimum overlap between patches (uses self.min_overlap if None)
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            device: Device for computation
            use_orthogonal_reg: Whether to use orthogonal regularization (overrides init setting)
            orthogonal_reg_weight: Weight for orthogonal regularization (overrides init setting)
            batch_size: Batch size for training (overrides init setting, 0 = full batch)
            center_patches: Whether to center patches (overrides init setting)
            
        Returns:
            Self for method chaining
            
        Raises:
            RuntimeError: If patch graph is not connected
        """
        # Override settings if provided
        if use_orthogonal_reg is not None:
            self.use_orthogonal_reg = use_orthogonal_reg
        if orthogonal_reg_weight is not None:
            self.orthogonal_reg_weight = orthogonal_reg_weight
        if batch_size is not None:
            self.batch_size = batch_size
        if center_patches is not None:
            self.center_patches = center_patches
            
        # Center patches if requested
        if self.center_patches:
            if self.verbose:
                print("Centering patches...")
            centered_patches = self._center_patches(patches)
        else:
            centered_patches = patches
            
        # Register patches (use centered patches for overlap computation)
        self._register_patches(centered_patches, min_overlap)
        
        if self.verbose:
            print(f"Aligning {self.n_patches} patches with {self.dim}D coordinates")
            if self.use_orthogonal_reg:
                print(f"Using orthogonal regularization with weight: {self.orthogonal_reg_weight}")
            else:
                print("Orthogonal regularization disabled")
        
        # Create patch graph
        self.patch_graph = self._create_patch_graph(self.patches, self.min_overlap)
        
        # Initialize model (disable bias when centering patches)
        use_bias = not self.center_patches
        self.model = GeoModel(device=device, n_patches=self.n_patches, dim=self.dim, use_bias=use_bias)
        
        # Generate training data
        self.training_data = self._generate_training_data(self.patches, device=device)
        
        if len(self.training_data) == 0:
            raise RuntimeError("No training data generated. Check patch overlaps and min_overlap setting.")
        
        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.loss_history = []
        
        if self.verbose:
            batch_mode = "full-batch" if self.batch_size == 0 else f"minibatch (size={self.batch_size})"
            print(f"Starting training: {num_epochs} epochs, lr={learning_rate}, {batch_mode}")
        
        for epoch in range(num_epochs):
            if self.batch_size == 0:
                # Full-batch training (current implementation)
                optimizer.zero_grad()
                loss = self._compute_loss(self.model, self.training_data)
                loss.backward()
                optimizer.step()
                self.loss_history.append(loss.item())
            else:
                # Optimized minibatch training with gradient accumulation
                optimizer.zero_grad()
                epoch_loss = 0.0
                num_samples = 0
                
                for (i, j), (x_i, x_j) in self.training_data.items():
                    # Process data in batches but accumulate gradients
                    n_batch_samples = x_i.shape[0]
                    for start_idx in range(0, n_batch_samples, self.batch_size):
                        end_idx = min(start_idx + self.batch_size, n_batch_samples)
                        
                        # Extract minibatch
                        batch_x_i = x_i[start_idx:end_idx]
                        batch_x_j = x_j[start_idx:end_idx]
                        
                        # Forward pass on minibatch
                        y_i, y_j = self.model.forward_batch(i, j, batch_x_i, batch_x_j)
                        
                        # Compute loss for this minibatch
                        mse_loss = F.mse_loss(y_i, y_j, reduction='sum')
                        epoch_loss += mse_loss.item()
                        
                        # Backward pass (accumulate gradients)
                        mse_loss.backward()
                        
                        num_samples += batch_x_i.shape[0]
                
                # Add regularization loss once per epoch
                if self.use_orthogonal_reg:
                    reg_loss = self._compute_orthogonal_regularization(self.model)
                    total_reg_loss = self.orthogonal_reg_weight * reg_loss
                    total_reg_loss.backward()
                    epoch_loss += total_reg_loss.item()
                
                # Single optimizer step per epoch
                optimizer.step()
                self.loss_history.append(epoch_loss)
            
            # Log progress
            if self.verbose and (epoch % max(1, num_epochs // 10) == 0 or epoch == num_epochs - 1):
                print(f"  Epoch {epoch:4d}: loss = {self.loss_history[-1]:.6f}")
        
        # Extract learned transformations
        self._extract_transformations()
        
        # Apply transformations to patches
        self._apply_transformations()
        
        # Compute final embedding
        self._aligned_embedding = self.mean_embedding()
        
        if self.verbose:
            print(f"Training completed. Final loss: {self.loss_history[-1]:.6f}")
        
        return self

    def _extract_transformations(self):
        """Extract weight matrices and biases from the trained model."""
        self.rotations = []
        self.shifts = []
        
        for i in range(self.n_patches):
            # Extract weight (rotation)
            weight = self.model.transformations[i].weight.detach().cpu().numpy()
            self.rotations.append(weight)
            
            # Extract bias (translation) if model uses bias, otherwise zero
            if self.model.use_bias:
                bias = self.model.transformations[i].bias.detach().cpu().numpy()
            else:
                bias = torch.zeros(self.dim).numpy()
            self.shifts.append(bias)

    def _apply_transformations(self):
        """Apply learned transformations to patch coordinates."""
        for i, patch in enumerate(self.patches):
            if self.center_patches:
                # For centered patches: we trained on centered data, so only rotation is learned
                # Apply rotation around the patch center
                center = self.patch_centers[i]
                centered_coords = patch.coordinates - center
                rotated_coords = centered_coords @ self.rotations[i].T
                patch.coordinates = rotated_coords + center
            else:
                # Original behavior: apply full affine transformation
                patch.coordinates = patch.coordinates @ self.rotations[i].T + self.shifts[i]