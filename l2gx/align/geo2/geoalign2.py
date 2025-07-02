"""
Redesigned geometric alignment implementation.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional
import copy

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
                 use_orthogonal_reg: bool = True, orthogonal_reg_weight: float = 100.0,
                 batch_size: int = 512, center_patches: bool = True, use_bfs_training: bool = True):
        """
        Initialize the geometric alignment problem.
        
        Args:
            verbose: Whether to print debug information
            min_overlap: Minimum overlap required between patches
            use_orthogonal_reg: Whether to use orthogonal regularization
            orthogonal_reg_weight: Weight for orthogonal regularization term
            batch_size: Batch size for training (0 = full batch)
            center_patches: Whether to center patches before applying transformations
            use_bfs_training: Whether to use breadth-first spanning tree training approach
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
        self.use_bfs_training = use_bfs_training
        self.patch_centers = {}
        self.bfs_tree = None
        self.bfs_levels = []

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

    def _compute_bfs_spanning_tree(self) -> dict:
        """
        Compute a breadth-first spanning tree of the patch graph.
        
        Returns:
            Dictionary with BFS tree structure containing:
            - 'tree_edges': List of (parent, child) edges in the spanning tree
            - 'levels': List of lists, each containing patch indices at that BFS level
            - 'parent_map': Dictionary mapping child patch index to parent patch index
        """
        if self.patch_graph is None:
            raise RuntimeError("Patch graph must be created before computing BFS tree")
            
        # Build adjacency list from patch graph
        adj_list = {i: [] for i in range(self.n_patches)}
        edge_index = self.patch_graph.edge_index.cpu().numpy()
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src != dst:  # Avoid self-loops
                adj_list[src].append(dst)
        
        # BFS starting from patch 0 (root)
        visited = set()
        queue = [0]
        visited.add(0)
        
        levels = [[0]]  # Level 0 contains only the root patch
        parent_map = {0: None}  # Root has no parent
        tree_edges = []
        
        current_level = 0
        
        while queue:
            # Process all nodes at current level
            next_level = []
            
            for _ in range(len(queue)):
                current_node = queue.pop(0)
                
                # Explore neighbors
                for neighbor in adj_list[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        next_level.append(neighbor)
                        parent_map[neighbor] = current_node
                        tree_edges.append((current_node, neighbor))
            
            if next_level:
                levels.append(next_level)
                current_level += 1
        
        if self.verbose:
            print(f"BFS spanning tree computed: {len(levels)} levels")
            for level, patches in enumerate(levels):
                print(f"  Level {level}: patches {patches}")
        
        return {
            'tree_edges': tree_edges,
            'levels': levels,
            'parent_map': parent_map
        }

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

    def _compute_orthogonal_regularization(self, model: GeoModel, active_patches: set = None) -> torch.Tensor:
        """
        Compute orthogonal regularization loss: ||W @ W.T - I||²_F
        
        Args:
            model: The GeoModel
            active_patches: Set of patch indices to include in regularization (None = all)
            
        Returns:
            Orthogonal regularization loss (scalar tensor)
        """
        reg_loss = 0.0
        
        for i, transformation in enumerate(model.transformations):
            # Skip the fixed first transformation (identity)
            if i == 0:
                continue
                
            # Skip if patch is not in active set (for BFS training)
            if active_patches is not None and i not in active_patches:
                continue
                
            W = transformation.weight  # Shape: (dim, dim)
            
            # Compute W @ W.T
            WWT = torch.mm(W, W.T)
            
            # Compute ||W @ W.T - I||²_F
            identity = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
            diff = WWT - identity
            reg_loss += torch.sum(diff * diff)  # Frobenius norm squared
        
        return reg_loss

    def _train_bfs_levels(self, num_epochs: int, learning_rate: float, optimizer) -> None:
        """
        Train patches individually using BFS spanning tree.
        Each patch is aligned to its parent patch one at a time.
        
        Args:
            num_epochs: Number of epochs to train each patch
            learning_rate: Learning rate for optimization
            optimizer: Optimizer instance
        """
        levels = self.bfs_tree['levels']
        parent_map = self.bfs_tree['parent_map']
        
        if self.verbose:
            batch_mode = "full-batch" if self.batch_size == 0 else f"minibatch (size={self.batch_size})"
            print(f"Starting BFS training: {len(levels)} levels, {num_epochs} epochs per patch, lr={learning_rate}, {batch_mode}")
        
        # Train each level sequentially, and within each level, train each patch individually
        for level_idx, level_patches in enumerate(levels):
            if level_idx == 0:
                # Skip level 0 (root patch) - it remains identity
                if self.verbose:
                    print(f"Level {level_idx}: root patch {level_patches[0]} (identity transformation)")
                continue
                
            if self.verbose:
                print(f"Level {level_idx}: training patches {level_patches}")
            
            # Train each patch in this level individually
            for patch_idx in level_patches:
                parent_idx = parent_map[patch_idx]
                
                if self.verbose:
                    print(f"  Training patch {patch_idx} -> parent {parent_idx}")
                
                # Find training data for this specific patch-parent pair
                patch_training_data = {}
                for (i, j), data in self.training_data.items():
                    if (i == patch_idx and j == parent_idx) or (i == parent_idx and j == patch_idx):
                        patch_training_data[(i, j)] = data
                
                if len(patch_training_data) == 0:
                    if self.verbose:
                        print(f"    No training data for patch {patch_idx} -> parent {parent_idx}, skipping")
                    continue
                
                if self.verbose:
                    overlap_size = sum(data[0].shape[0] for data in patch_training_data.values())
                    print(f"    Training with {len(patch_training_data)} patch pairs, {overlap_size} overlap points")
                    
                    # Debug: show initial transformation matrix
                    initial_weight = self.model.transformations[patch_idx].weight.detach().cpu().numpy()
                    print(f"    Initial transformation matrix for patch {patch_idx}:")
                    print(f"      {initial_weight}")
                
                # Train this specific patch for specified number of epochs
                for epoch in range(num_epochs):
                    optimizer.zero_grad()
                    
                    # Compute loss only for this patch-parent pair
                    loss = self._compute_loss(self.model, patch_training_data, {patch_idx})
                    loss.backward()
                    
                    # Zero out gradients for all patches except current patch
                    self._mask_gradients_for_patch(patch_idx)
                    
                    optimizer.step()
                    
                    # Safeguard: prevent transformations from becoming degenerate or flipping orientation
                    with torch.no_grad():
                        W = self.model.transformations[patch_idx].weight
                        det = torch.det(W)
                        
                        # If determinant is too small, regularize towards identity
                        if abs(det) < 0.1:
                            if self.verbose and epoch % max(1, num_epochs // 3) == 0:
                                print(f"        Warning: Determinant {det:.6f} is small, regularizing towards identity")
                            # Push weight matrix towards identity
                            identity = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
                            W.data = 0.9 * W.data + 0.1 * identity
                        
                        # If determinant became negative, flip orientation back to positive
                        elif det < 0:
                            if self.verbose and epoch % max(1, num_epochs // 3) == 0:
                                print(f"        Warning: Determinant {det:.6f} is negative, flipping orientation")
                            # Flip the first column to make determinant positive
                            W.data[:, 0] *= -1
                    
                    self.loss_history.append(loss.item())
                    
                    # Log progress (less frequent for individual patches)
                    if self.verbose and (epoch % max(1, num_epochs // 3) == 0 or epoch == num_epochs - 1):
                        print(f"      Patch {patch_idx}, Epoch {epoch:3d}: loss = {loss.item():.6f}")
                
                if self.verbose:
                    # Debug: show final transformation matrix and orthogonality measure
                    final_weight = self.model.transformations[patch_idx].weight.detach().cpu().numpy()
                    print(f"    Final transformation matrix for patch {patch_idx}:")
                    print(f"      {final_weight}")
                    
                    # Measure how orthogonal the matrix is
                    WWT = final_weight @ final_weight.T
                    identity = torch.eye(final_weight.shape[0]).numpy()
                    orthogonality_error = torch.norm(torch.tensor(WWT) - torch.tensor(identity)).item()
                    print(f"    Orthogonality error: {orthogonality_error:.6f}")
                    
                    # Measure determinant (should be close to ±1 for orthogonal matrices)
                    det = torch.det(torch.tensor(final_weight)).item()
                    print(f"    Determinant: {det:.6f}")

    def _mask_gradients_for_level(self, current_level_patches: list) -> None:
        """
        Zero out gradients for all patches except those in the current level.
        
        Args:
            current_level_patches: List of patch indices in the current level
        """
        current_level_set = set(current_level_patches)
        
        for i, transformation in enumerate(self.model.transformations):
            if i not in current_level_set:
                # Zero out gradients for patches not in current level
                if transformation.weight.grad is not None:
                    transformation.weight.grad.zero_()
                if transformation.bias is not None and transformation.bias.grad is not None:
                    transformation.bias.grad.zero_()

    def _mask_gradients_for_patch(self, current_patch: int) -> None:
        """
        Zero out gradients for all patches except the current patch.
        
        Args:
            current_patch: Index of the patch to keep gradients for
        """
        for i, transformation in enumerate(self.model.transformations):
            if i != current_patch:
                # Zero out gradients for patches not being trained
                if transformation.weight.grad is not None:
                    transformation.weight.grad.zero_()
                if transformation.bias is not None and transformation.bias.grad is not None:
                    transformation.bias.grad.zero_()

    def _verify_patch_overlaps(self) -> None:
        """
        Verify that patches still have meaningful overlaps after transformation.
        """
        print("Verifying patch overlaps after transformation:")
        
        total_overlaps = 0
        preserved_overlaps = 0
        
        for (i, j), overlap_nodes in self.overlaps.items():
            # Get transformed coordinates for overlapping nodes
            coords_i = self.patches[i].get_coordinates(overlap_nodes)
            coords_j = self.patches[j].get_coordinates(overlap_nodes)
            
            # Compute distances between corresponding points
            distances = torch.norm(torch.tensor(coords_i) - torch.tensor(coords_j), dim=1)
            mean_distance = distances.mean().item()
            max_distance = distances.max().item()
            
            # Consider overlap preserved if mean distance is reasonable
            overlap_preserved = mean_distance < 1.0  # Adjust threshold as needed
            
            total_overlaps += 1
            if overlap_preserved:
                preserved_overlaps += 1
            
            print(f"  Patches {i}-{j}: {len(overlap_nodes)} nodes, "
                  f"mean_dist={mean_distance:.4f}, max_dist={max_distance:.4f}, "
                  f"preserved={'Yes' if overlap_preserved else 'No'}")
        
        preservation_rate = preserved_overlaps / total_overlaps if total_overlaps > 0 else 0
        print(f"Overlap preservation rate: {preserved_overlaps}/{total_overlaps} ({preservation_rate:.1%})")

    def _compute_loss(self, model: GeoModel, training_data: dict, active_patches: set = None) -> torch.Tensor:
        """
        Compute the total loss over all training examples using batched operations.
        
        Args:
            model: The GeoModel
            training_data: Dictionary of batched training data by patch pair
            active_patches: Set of patch indices to include in loss computation (None = all)
            
        Returns:
            Total loss (scalar tensor)
        """
        # Compute MSE loss using batched operations
        mse_loss = 0.0
        
        for (i, j), (x_i, x_j) in training_data.items():
            # Skip if neither patch is in active set (for BFS training)
            if active_patches is not None and i not in active_patches and j not in active_patches:
                continue
                
            # Batched forward pass
            y_i, y_j = model.forward_batch(i, j, x_i, x_j)
            
            # Batched squared difference loss
            loss = F.mse_loss(y_i, y_j, reduction='sum')
            mse_loss += loss
        
        # Add orthogonal regularization if enabled (only for active patches)
        total_loss = mse_loss
        if self.use_orthogonal_reg:
            reg_loss = self._compute_orthogonal_regularization(model, active_patches)
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
        center_patches: Optional[bool] = None,
        use_bfs_training: Optional[bool] = None
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
        if use_bfs_training is not None:
            self.use_bfs_training = use_bfs_training
            
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
        
        if self.use_bfs_training:
            # BFS-based training: train level by level
            self.bfs_tree = self._compute_bfs_spanning_tree()
            self._train_bfs_levels(num_epochs, learning_rate, optimizer)
        else:
            # Original training approach: train all patches simultaneously
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
            
            # Check if patches still overlap after transformation
            self._verify_patch_overlaps()
        
        return self

    def align_patches_iteratively(
        self, 
        patches: list[Patch], 
        num_iterations: int = 2,
        min_overlap: Optional[int] = None,
        num_epochs: int = 1000,
        learning_rate: float = 0.01,
        device: str = "cpu",
        use_orthogonal_reg: Optional[bool] = None,
        orthogonal_reg_weight: Optional[float] = None,
        batch_size: Optional[int] = None,
        center_patches: Optional[bool] = None,
        use_bfs_training: Optional[bool] = None
    ) -> 'GeoAlignmentProblem2':
        """
        Perform iterative alignment by running the alignment procedure multiple times.
        Each iteration uses the result of the previous iteration as starting point.
        
        Args:
            patches: List of patches to align
            num_iterations: Number of alignment iterations to perform
            min_overlap: Minimum overlap between patches (uses self.min_overlap if None)
            num_epochs: Number of training epochs per iteration
            learning_rate: Learning rate for optimization
            device: Device for computation
            use_orthogonal_reg: Whether to use orthogonal regularization (overrides init setting)
            orthogonal_reg_weight: Weight for orthogonal regularization (overrides init setting)
            batch_size: Batch size for training (overrides init setting, 0 = full batch)
            center_patches: Whether to center patches (overrides init setting)
            use_bfs_training: Whether to use BFS training (overrides init setting)
            
        Returns:
            Self for method chaining
            
        Raises:
            RuntimeError: If patch graph is not connected in any iteration
        """
        if self.verbose:
            print(f"Starting iterative alignment: {num_iterations} iterations")
        
        # Store original initialization parameters for reset
        original_params = {
            'verbose': self.verbose,
            'min_overlap': self.min_overlap,
            'use_orthogonal_reg': self.use_orthogonal_reg,
            'orthogonal_reg_weight': self.orthogonal_reg_weight,
            'batch_size': self.batch_size,
            'center_patches': self.center_patches,
            'use_bfs_training': self.use_bfs_training
        }
        
        current_patches = [self._copy_patch(patch) for patch in patches]  # Deep copy to avoid modifying originals
        
        for iteration in range(num_iterations):
            if self.verbose:
                print(f"\n=== Alignment Iteration {iteration + 1}/{num_iterations} ===")
            
            # Run one full alignment iteration
            self.align_patches(
                current_patches,
                min_overlap=min_overlap,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                device=device,
                use_orthogonal_reg=use_orthogonal_reg,
                orthogonal_reg_weight=orthogonal_reg_weight,
                batch_size=batch_size,
                center_patches=center_patches,
                use_bfs_training=use_bfs_training
            )
            
            if iteration < num_iterations - 1:  # Don't reset after the last iteration
                # Get the transformed patches for next iteration
                current_patches = [self._copy_patch(patch) for patch in self.patches]
                
                if self.verbose:
                    print(f"Iteration {iteration + 1} completed. Preparing for next iteration...")
                    # Show some statistics about current alignment quality
                    if hasattr(self, 'loss_history') and self.loss_history:
                        print(f"  Final loss: {self.loss_history[-1]:.6f}")
                
                # Reset the aligner state for next iteration
                self._reset_for_next_iteration(original_params)
        
        if self.verbose:
            print(f"\nIterative alignment completed after {num_iterations} iterations")
            if hasattr(self, 'loss_history') and self.loss_history:
                print(f"Final loss: {self.loss_history[-1]:.6f}")
        
        return self

    def _reset_for_next_iteration(self, original_params: dict) -> None:
        """
        Reset the aligner state for the next iteration.
        
        Args:
            original_params: Dictionary of original initialization parameters
        """
        # Reset all internal state
        self.patch_graph = None
        self.model = None
        self.training_data = []
        self.loss_history = []
        self.patch_centers = {}
        self.bfs_tree = None
        self.bfs_levels = []
        self.patches = []
        self.overlaps = {}
        
        # Reset transformations if they exist
        if hasattr(self, 'rotations'):
            self.rotations = []
        if hasattr(self, 'shifts'):
            self.shifts = []
        if hasattr(self, '_aligned_embedding'):
            self._aligned_embedding = None
        
        # Restore original parameters
        self.verbose = original_params['verbose']
        self.min_overlap = original_params['min_overlap']
        self.use_orthogonal_reg = original_params['use_orthogonal_reg']
        self.orthogonal_reg_weight = original_params['orthogonal_reg_weight']
        self.batch_size = original_params['batch_size']
        self.center_patches = original_params['center_patches']
        self.use_bfs_training = original_params['use_bfs_training']

    def _copy_patch(self, patch: Patch) -> Patch:
        """
        Create a deep copy of a patch.
        
        Args:
            patch: Patch to copy
            
        Returns:
            Deep copy of the patch
        """
        # Create a new patch with copied nodes and coordinates
        new_patch = Patch(
            nodes=patch.nodes.copy(),
            coordinates=patch.coordinates.copy() if patch.coordinates is not None else None
        )
        return new_patch

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