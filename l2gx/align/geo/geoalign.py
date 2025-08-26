"""
Geometric alignment implementation.
"""


import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from l2gx.align.alignment import (
    AlignmentProblem,
    nearest_orthogonal,
    relative_orthogonal_transform,
)
from l2gx.align.nla import synchronise
from l2gx.align.registry import register_aligner
from l2gx.graphs.tgraph import TGraph
from l2gx.patch import Patch

from .models import GeoModel, GeoModelOrthogonal


@register_aligner("geo")
class GeoAlignmentProblem(AlignmentProblem):
    """
    Redesigned geometric alignment using patch graphs and simplified model.
    """

    def __init__(
        self,
        verbose: bool = False,
        use_scale: bool = True,
        method: str = "standard",
        orthogonal_reg_weight: float = 100.0,
        batch_size: int = 512,
        use_bfs_training: bool = True,
        device: str = "cpu",
        patience: int = 20,
        tolerance: float = 1e-8,
        use_randomized_init: bool = False,
        randomized_method: str = "randomized",
    ):
        """
        Initialize the geometric alignment problem.

        Args:
            verbose: Whether to print debug information
            use_scale: Whether to scale patches before applying transformations
            method: Model type - "standard" (with regularization) or "orthogonal" (parametric)
            orthogonal_reg_weight: Weight for orthogonal regularization term (only for standard method)
            batch_size: Batch size for training (0 = full batch)
            use_bfs_training: Whether to use breadth-first spanning tree training approach
            device: Device for computation
            patience: Number of epochs to wait for improvement before early stopping
            tolerance: Minimum loss threshold for early stopping
            use_randomized_init: Whether to use randomized synchronization for initial rotations
            randomized_method: Method for randomized synchronization ("randomized", "adaptive", "sparse_aware", "standard")
        """
        super().__init__(verbose=verbose)
        self.model = None
        self.training_data = []
        self.loss_history = []
        self.use_scale = use_scale
        self.method = method
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.batch_size = batch_size
        self.use_bfs_training = use_bfs_training
        self.patience = int(patience)
        self.tolerance = float(tolerance)
        self.patch_centers = {}
        self.bfs_tree = None
        self.bfs_levels = []
        self.device = device
        self.num_epochs = 1000
        self.learning_rate = 0.01
        self.best_loss = float("inf")
        self.best_model_state = None
        self.use_randomized_init = use_randomized_init
        self.randomized_method = randomized_method
        self.initial_rotations = None

    def _compute_bfs_spanning_tree(self, patch_graph: TGraph) -> dict:
        """
        Compute a breadth-first spanning tree of the patch graph.
        As the patch graph is typically small, efficiency is not a concern.

        Args:
            patch_graph: The patch graph to compute BFS tree from

        Returns:
            Dictionary with BFS tree structure containing:
            - 'tree_edges': list of (parent, child) edges in the spanning tree
            - 'levels': list of lists, each containing patch indices at that BFS level
            - 'parent_map': Dictionary mapping child patch index to parent patch index
        """
        # Build adjacency list from patch graph
        adj_list = {i: [] for i in range(self.n_patches)}
        edge_index = patch_graph.edge_index.cpu().numpy()

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

                # Explore neighbours
                for neighbour in adj_list[current_node]:
                    if neighbour not in visited:
                        visited.add(neighbour)
                        queue.append(neighbour)
                        next_level.append(neighbour)
                        parent_map[neighbour] = current_node
                        tree_edges.append((current_node, neighbour))

            if next_level:
                levels.append(next_level)
                current_level += 1

        if self.verbose:
            print(f"BFS spanning tree computed: {len(levels)} levels")
            for level, patches in enumerate(levels):
                print(f"  Level {level}: patches {patches}")

        return {"tree_edges": tree_edges, "levels": levels, "parent_map": parent_map}

    def _center_patches(self, patches: list[Patch]) -> list[Patch]:
        """
        Center patches by subtracting their mean coordinates.

        Args:
            patches: list of patches to center

        Returns:
            list of centered patches

        Note: This method is kept for backward compatibility but is no longer
        used in the main alignment workflow. We now use L2G-style translation sync instead.
        """

        centered_patches = []
        self.patch_centers = {}

        for i, patch in enumerate(patches):
            center = patch.coordinates.mean(axis=0)
            self.patch_centers[i] = center
            centered_coords = patch.coordinates - center
            centered_patch = Patch(patch.nodes.copy(), centered_coords)
            centered_patches.append(centered_patch)

            if self.verbose:
                print(f"  Patch {i}: centered at {center}")

        return centered_patches

    def _should_early_stop(self, epoch: int, loss_history: list) -> bool:
        """
        Check if training should stop early based on patience and tolerance.

        Args:
            epoch: Current epoch number
            loss_history: list of loss values

        Returns:
            True if early stopping should be triggered, False otherwise
        """
        if len(loss_history) == 0:
            return False

        current_loss = loss_history[-1]

        # Check tolerance condition
        if current_loss <= self.tolerance:
            if self.verbose:
                print(
                    f"    Early stopping: Loss {current_loss:.2e} below tolerance {self.tolerance:.2e}"
                )
            return True

        # Check patience condition
        if len(loss_history) >= self.patience:
            # Look back 'patience' epochs to check for improvement
            lookback_loss = loss_history[-self.patience]
            improvement = (lookback_loss - current_loss) / max(lookback_loss, 1e-10)

            # Consider improvement significant if it's more than 1e-5 relative improvement
            min_improvement = 1e-5
            if improvement < min_improvement:
                if self.verbose:
                    print(
                        f"    Early stopping: No improvement for {self.patience} epochs"
                    )
                    print(
                        f"    Current loss: {current_loss:.6f}, {self.patience} epochs ago: {lookback_loss:.6f}"
                    )
                return True

        return False

    def _save_best_model(self, current_loss: float):
        """
        Save the current model state if it's the best so far.

        Args:
            current_loss: Current training loss
        """
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            if self.model is not None:
                # Save a deep copy of the model state
                import copy

                self.best_model_state = copy.deepcopy(self.model.state_dict())

    def _restore_best_model(self):
        """
        Restore the model to the best state encountered during training.
        """
        if self.best_model_state is not None and self.model is not None:
            self.model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"    Restored best model with loss: {self.best_loss:.6f}")

    def _generate_training_data(self) -> dict:
        """
        Generate pre-batched training data from patch overlaps.

        For each patch pair (i,j), creates:
        - x_i: coordinates of patch i corresponding to nodes in intersection of patches i and j
        - x_j: coordinates of patch j corresponding to nodes in intersection of patches i and j

        Args:
            patches: list of patches
            device: Device to place tensors on

        Returns:
            Dictionary with pre-batched training data by patch pair
        """
        training_data = {}
        total_examples = 0

        for (i, j), overlap_nodes in self.patch_overlap.items():
            # Get coordinates for overlapping nodes in both patches
            # x_i: coordinates from patch i for nodes in intersection of patches i and j
            coords_i = self.patches[i].get_coordinates(overlap_nodes)
            # x_j: coordinates from patch j for nodes in intersection of patches i and j
            coords_j = self.patches[j].get_coordinates(overlap_nodes)

            # Center the coordinates temporarily (like L2G does in _cov_svd)
            # This ensures the model receives centered data while keeping patches uncentered
            coords_i = coords_i - coords_i.mean(axis=0)
            coords_j = coords_j - coords_j.mean(axis=0)

            # Create pre-batched tensors for this patch pair
            # These will be passed directly to forward_batch method
            x_i = torch.tensor(
                coords_i, dtype=torch.float32, device=self.device
            )  # Shape: (n_overlap, dim)
            x_j = torch.tensor(
                coords_j, dtype=torch.float32, device=self.device
            )  # Shape: (n_overlap, dim)

            training_data[(i, j)] = (x_i, x_j)
            total_examples += len(overlap_nodes)

            if self.verbose:
                print(f"  Batch ({i}, {j}): {len(overlap_nodes)} overlapping nodes")

        if self.verbose:
            print(
                f"Generated {total_examples} training examples in {len(training_data)} pre-batches"
            )
            print("Each batch contains x_i and x_j for nodes in patch intersection")

        return training_data

    def _compute_orthogonal_regularization(
        self, model: GeoModel, active_patches: set = None, single_patch: int = None
    ) -> torch.Tensor:
        """
        Compute orthogonal regularization loss: ||W @ W.T - I||²_F

        Args:
            model: The GeoModel
            active_patches: set of patch indices to include in regularization (None = all)
            single_patch: If specified, only regularize this specific patch index

        Returns:
            Orthogonal regularization loss (scalar tensor)
        """
        reg_loss = 0.0

        for i, transformation in enumerate(model.transformations):
            # Skip the fixed first transformation (identity)
            if i == 0:
                continue

            # If single_patch is specified, only regularize that patch
            if single_patch is not None:
                if i != single_patch:
                    continue
            # Otherwise, skip if patch is not in active set (for BFS training)
            elif active_patches is not None and i not in active_patches:
                continue

            W = transformation.weight  # Shape: (dim, dim)

            # Compute W @ W.T
            WWT = torch.mm(W, W.T)

            # Compute ||W @ W.T - I||²_F
            identity = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
            diff = WWT - identity
            reg_loss += torch.sum(diff * diff)  # Frobenius norm squared

        return reg_loss

    def _compute_orthogonality_metrics(self, model: GeoModel) -> dict:
        """
        Compute orthogonality metrics for all transformation matrices.

        Args:
            model: The GeoModel

        Returns:
            Dictionary with orthogonality metrics
        """
        metrics = {
            "orthogonality_errors": [],
            "determinants": [],
            "condition_numbers": [],
        }

        for i, transformation in enumerate(model.transformations):
            if i == 0:  # Skip fixed identity transformation
                continue

            W = transformation.weight.detach().cpu().numpy()

            # Compute orthogonality error: ||W @ W.T - I||_F
            WWT = W @ W.T
            identity = np.eye(W.shape[0])
            orth_error = np.linalg.norm(WWT - identity, "fro")
            metrics["orthogonality_errors"].append(orth_error)

            # Compute determinant (should be ±1 for orthogonal matrices)
            det = np.linalg.det(W)
            metrics["determinants"].append(det)

            # Compute condition number (should be 1 for orthogonal matrices)
            cond_num = np.linalg.cond(W)
            metrics["condition_numbers"].append(cond_num)

        return metrics

    def _reinitialize_model(self):
        """
        Reinitialize the model parameters while keeping the same architecture.
        This allows escaping local minima by starting optimization from a new point.
        """
        if self.verbose:
            print("    Reinitializing model parameters...")

        # Store current model settings
        device = self.model.device
        n_patches = self.model.n_patches
        dim = self.model.dim
        use_bias = self.model.use_bias

        # Create new model with same settings
        if self.method == "orthogonal":
            self.model = GeoModelOrthogonal(
                device=device, n_patches=n_patches, dim=dim, use_bias=use_bias
            )
        else:
            self.model = GeoModel(
                device=device, n_patches=n_patches, dim=dim, use_bias=use_bias
            )

        if self.verbose:
            print("    Model reinitialized successfully")

    def _update_patches_from_model(self):
        """
        Update patch coordinates based on current model state.
        This preserves the current alignment progress before restarting.
        """
        if self.verbose:
            print("    Updating patches from current model state...")

        # Extract current transformations
        current_rotations = []
        current_shifts = []

        for i in range(self.n_patches):
            # Extract weight (rotation)
            weight = self.model.transformations[i].weight.detach().cpu().numpy()
            current_rotations.append(weight)

            # Extract bias or compute translations
            # Will be computed using L2G method
            current_shifts.append(np.zeros(self.dim))

        # Apply current transformations to patches
        for i, patch in enumerate(self.patches):
            # Apply rotation
            patch.coordinates = patch.coordinates @ current_rotations[i].T

            # Apply translation using L2G-style synchronized translations
            if hasattr(self, "patch_overlap") and self.patch_overlap:
                translations = self.calc_synchronised_translations()
                patch.coordinates += translations[i]
                current_shifts[i] = translations[i]
            else:
                # Fallback if patch_overlap not available
                patch.coordinates += current_shifts[i]

        # Update stored transformations
        self.rotations = current_rotations
        self.shifts = current_shifts

        if self.verbose:
            print("    Patches updated successfully")

    def _compute_initial_rotations(self) -> list:
        """
        Compute initial rotation matrices using randomized synchronization.

        This method computes the synchronized rotations using the L2G approach
        with the specified randomized method, which can be used as initial
        weights for the neural network optimization.

        Returns:
            list of rotation matrices (numpy arrays)
        """
        # For initialization, we can use faster computation with fewer iterations
        # since we just need a rough initial estimate
        rots = self._transform_matrix(
            relative_orthogonal_transform, self.dim, symmetric_weights=True
        )

        # Only pass n_iter for randomized methods that support it
        if self.randomized_method in ["randomized", "adaptive"]:
            vecs = synchronise(
                rots,
                blocksize=self.dim,
                symmetric=True,
                method=self.randomized_method,
                n_iter=2,
                verbose=self.verbose,
            )
        else:
            # For "standard" and "sparse_aware" methods that don't accept n_iter
            vecs = synchronise(
                rots,
                blocksize=self.dim,
                symmetric=True,
                method=self.randomized_method,
                verbose=self.verbose,
            )

        # Apply nearest orthogonal projection
        for mat in vecs:
            mat[:] = nearest_orthogonal(mat)

        rotations = vecs

        # Convert to list of numpy arrays for easier handling
        rotation_list = []
        for i in range(self.n_patches):
            rotation_list.append(rotations[i])

        return rotation_list

    def _set_model_initial_rotations(self) -> None:
        """
        set the model's initial weights to the pre-computed rotations.

        This method initializes the transformation matrices in the GeoModel
        with the rotations computed by randomized synchronization.
        """
        if self.initial_rotations is None:
            return

        with torch.no_grad():
            for i, rotation in enumerate(self.initial_rotations):
                # if i == 0:
                # Keep the first transformation as identity (reference patch)
                # continue

                # Convert numpy array to torch tensor
                # Note: L2G applies rot.T, but geo model stores the rotation directly
                # and applies it as coordinates @ weight.T, so we need to transpose here
                rotation_tensor = torch.tensor(
                    rotation.T, dtype=torch.float32, device=self.device
                )

                # Handle parametric vs non-parametric models differently
                layer = self.model.transformations[i]

                # For parametric models, we need to access the original weight
                if (
                    hasattr(layer, "parametrizations")
                    and "weight" in layer.parametrizations
                ):
                    # This is a parametrized layer - access the original weight
                    with torch.no_grad():
                        # The parametrization will automatically make it orthogonal
                        layer.parametrizations.weight.original.copy_(rotation_tensor)
                elif hasattr(layer, "weight"):
                    # Non-parametrized layer - set weight directly
                    layer.weight.copy_(rotation_tensor)
                else:
                    if self.verbose:
                        print(
                            f"  Warning: Could not set initial rotation for patch {i}"
                        )

                if self.verbose:
                    # Check determinant and orthogonality
                    det = torch.det(rotation_tensor).item()
                    orth_error = torch.norm(
                        rotation_tensor @ rotation_tensor.T
                        - torch.eye(self.dim, device=self.device)
                    ).item()
                    print(
                        f"  Patch {i}: initial rotation det={det:.6f}, orthogonality error={orth_error:.6f}"
                    )

    def _verify_patch_overlaps(self) -> None:
        """
        Verify that patches still have meaningful overlaps after transformation.
        """
        print("Verifying patch overlaps after transformation:")

        total_overlaps = 0
        preserved_overlaps = 0

        for (i, j), overlap_nodes in self.patch_overlap.items():
            # Get transformed coordinates for overlapping nodes
            coords_i = self.patches[i].get_coordinates(overlap_nodes)
            coords_j = self.patches[j].get_coordinates(overlap_nodes)

            # Compute distances between corresponding points
            distances = torch.norm(
                torch.tensor(coords_i) - torch.tensor(coords_j), dim=1
            )
            mean_distance = distances.mean().item()
            max_distance = distances.max().item()

            # Consider overlap preserved if mean distance is reasonable
            overlap_preserved = mean_distance < 1.0  # Adjust threshold as needed

            total_overlaps += 1
            if overlap_preserved:
                preserved_overlaps += 1

            print(
                f"  Patches {i}-{j}: {len(overlap_nodes)} nodes, "
                f"mean_dist={mean_distance:.4f}, max_dist={max_distance:.4f}, "
                f"preserved={'Yes' if overlap_preserved else 'No'}"
            )

        preservation_rate = (
            preserved_overlaps / total_overlaps if total_overlaps > 0 else 0
        )
        print(
            f"Overlap preservation rate: {preserved_overlaps}/{total_overlaps} ({preservation_rate:.1%})"
        )

    def _compute_loss(
        self, model: GeoModel, training_data: dict, active_patches: set = None
    ) -> torch.Tensor:
        """
        Compute the total loss over all training examples using batched operations.

        Args:
            model: The GeoModel
            training_data: Dictionary of batched training data by patch pair
            active_patches: set of patch indices to include in loss computation (None = all)

        Returns:
            Total loss (scalar tensor)
        """
        # Compute MSE loss using batched operations
        mse_loss = 0.0

        for (i, j), (x_i, x_j) in training_data.items():
            # Skip if neither patch is in active set (for BFS training)
            if (
                active_patches is not None
                and i not in active_patches
                and j not in active_patches
            ):
                continue

            # Batched forward pass using pre-generated batches
            # x_i: coordinates from patch i for nodes in intersection of patches i and j
            # x_j: coordinates from patch j for nodes in intersection of patches i and j
            y_i, y_j = model.forward(i, j, x_i, x_j)

            # Batched squared difference loss
            loss = F.mse_loss(y_i, y_j, reduction="sum")
            mse_loss += loss

        # Add orthogonal regularization if enabled (only for standard method)
        total_loss = mse_loss
        if self.method == "standard" and self.orthogonal_reg_weight > 0:
            # Apply orthogonal regularization to ALL patches, not just active ones
            # This prevents untrained patches from drifting away from orthogonality
            # Only the MSE loss respects the BFS training strategy
            reg_loss = self._compute_orthogonal_regularization(
                model, active_patches=None
            )
            reg_weight = self.orthogonal_reg_weight
            total_loss += reg_weight * reg_loss

        return total_loss

    def rotate_patches(self) -> "GeoAlignmentProblem":
        """
        Align the rotation/reflection of all patches using geometric optimization.

        Uses the configuration parameters set in the object (from constructor or align_patches).

        Returns:
            Self for method chaining
        """

        if self.verbose:
            print(f"Aligning {self.n_patches} patches with {self.dim}D coordinates")
            if self.orthogonal_reg_weight > 0:
                print(
                    f"Using orthogonal regularization with weight: {self.orthogonal_reg_weight}"
                )
            else:
                print("Orthogonal regularization disabled")

        # Scale patches if requested - compute synchronized scales
        if self.use_scale:
            if self.verbose:
                print("Computing synchronized scales for patches...")
            self.scale_patches()  # Compute and apply synchronized scales automatically
            if self.verbose:
                print(f"Applied scale factors: {self.scales}")

        # Compute BFS spanning tree if BFS training is enabled
        if self.use_bfs_training:
            if self.verbose:
                print("Computing BFS spanning tree for level-wise training...")
            self.bfs_tree = self._compute_bfs_spanning_tree(self.patch_graph)
            self.bfs_levels = self.bfs_tree["levels"]

        # Note: We don't actually center the patches before training, just like L2G.
        # The training data is centered temporarily (like L2G's _cov_svd), but patches remain uncentered.
        # We use L2G-style translation computation after extracting rotations from model.

        # Debug: Check patch coordinate statistics after scaling
        if self.verbose and self.use_scale:
            print("Patch coordinate statistics after scaling:")
            for i, patch in enumerate(self.patches):
                coords = patch.coordinates
                mean_norm = np.mean(np.linalg.norm(coords, axis=1))
                std_norm = np.std(np.linalg.norm(coords, axis=1))
                print(
                    f"  Patch {i}: mean_norm={mean_norm:.4f}, std_norm={std_norm:.4f}, scale={self.scales[i]:.4f}"
                )

        # Compute initial rotations if requested
        if self.use_randomized_init:
            if self.verbose:
                print(
                    f"Computing initial rotations using {self.randomized_method} synchronization..."
                )
            self.initial_rotations = self._compute_initial_rotations()
            if self.verbose:
                print("Initial rotations computed successfully")

        # Initialize model (disable bias since training data is centered)
        if self.method == "orthogonal":
            self.model = GeoModelOrthogonal(
                device=self.device,
                n_patches=self.n_patches,
                dim=self.dim,
                use_bias=False,  # Training data is centered, so no bias needed
                initial_rotations=self.initial_rotations,
            )
        else:
            self.model = GeoModel(
                device=self.device,
                n_patches=self.n_patches,
                dim=self.dim,
                use_bias=False,
            )  # Training data is centered, so no bias needed

            # set initial rotations if available (only for standard model)
            if self.initial_rotations is not None:
                self._set_model_initial_rotations()

        # Generate training data
        self.training_data = self._generate_training_data()

        if len(self.training_data) == 0:
            raise RuntimeError(
                "No training data generated. Check patch overlaps and min_overlap setting."
            )

        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Train with SGD
        self._train_with_sgd(optimizer, self.num_epochs)

        # Extract and apply transformations
        self._extract_transformations()
        self._apply_transformations()

        if self.verbose:
            if hasattr(self, "loss_history") and self.loss_history:
                print(f"Training completed. Final loss: {self.loss_history[-1]:.6f}")

            # Check if patches still overlap after transformation
            self._verify_patch_overlaps()

        return self

    def align_patches(
        self,
        patch_graph: TGraph,
        use_scale: bool = True,
        num_epochs: int = 1000,
        learning_rate: float = 0.01,
        device: str = "cpu",
        use_randomized_init: bool | None = None,
        randomized_method: str | None = None,
    ) -> "GeoAlignmentProblem":
        """
        Align patches using the geometric approach.

        Args:
            patch_graph: Pre-computed patch graph with patches as node features and overlap information
            use_scale: Whether to perform scale synchronization
            num_epochs: Number of training epochs for rotate_patches
            learning_rate: Learning rate for optimization
            device: Device for computation
            use_randomized_init: Whether to use randomized initialization (overrides constructor setting)
            randomized_method: Randomized method to use (overrides constructor setting)

        Returns:
            Self for method chaining
        """
        self._register_patches(patch_graph)

        # set training parameters
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device

        # Override initialization settings if specified
        if use_randomized_init is not None:
            self.use_randomized_init = use_randomized_init
        if randomized_method is not None:
            self.randomized_method = randomized_method

        # Scale patches if requested - compute synchronized scales
        if use_scale:
            if self.verbose:
                print("Computing synchronized scales for patches...")
            self.scale_patches()  # Compute and apply synchronized scales automatically
            if self.verbose:
                print(f"Applied scale factors: {self.scales}")

        # Perform geometric alignment
        self.rotate_patches()
        # Apply L2G-style synchronized translations
        self.translate_patches()
        self._aligned_embedding = self.mean_embedding()

        return self

    def _train_with_sgd(self, optimizer, num_epochs):
        """Standard SGD training loop using BFS levels as batches."""
        if self.verbose:
            print(f"Starting SGD training for {num_epochs} epochs")
            if self.use_bfs_training:
                print(f"Using BFS training with {len(self.bfs_levels)} levels")
                for level, patches in enumerate(self.bfs_levels):
                    print(f"  Level {level}: patches {patches}")
            else:
                print("Using standard full-batch training")

        self.loss_history = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            if self.use_bfs_training:
                # Train level by level using BFS spanning tree
                for level in range(len(self.bfs_levels)):
                    if level == 0:
                        # Skip root level (no parent to align with)
                        continue

                    # Get active patches for this level (current level + parent levels)
                    active_patches = set()
                    for prev_level in range(level + 1):
                        active_patches.update(self.bfs_levels[prev_level])

                    # Compute loss for this level
                    optimizer.zero_grad()
                    loss = self._compute_loss(
                        self.model, self.training_data, active_patches
                    )

                    if loss.requires_grad:
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item()
            else:
                # Standard full-batch training
                optimizer.zero_grad()
                loss = self._compute_loss(self.model, self.training_data)

                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()

                epoch_loss = loss.item()

            self.loss_history.append(epoch_loss)

            # Save best model if current loss is better
            self._save_best_model(epoch_loss)

            # Check for early stopping
            if self._should_early_stop(epoch, self.loss_history):
                if self.verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

            # Print progress every 100 epochs
            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch + 1:4d}/{num_epochs}: Loss = {epoch_loss:.6f}")

                # Monitor orthogonality metrics
                metrics = self._compute_orthogonality_metrics(self.model)
                if metrics["orthogonality_errors"]:
                    avg_orth_error = np.mean(metrics["orthogonality_errors"])
                    max_orth_error = np.max(metrics["orthogonality_errors"])
                    avg_det = np.mean(np.abs(metrics["determinants"]))
                    avg_cond = np.mean(metrics["condition_numbers"])
                    print(
                        f"    Orthogonality: avg_error={avg_orth_error:.6f}, max_error={max_orth_error:.6f}, avg_|det|={avg_det:.6f}, avg_cond={avg_cond:.2f}"
                    )

        # Restore best model
        self._restore_best_model()

        if self.verbose:
            print(f"Training completed. Final loss: {self.loss_history[-1]:.6f}")
            print(
                f"Loss improvement: {self.loss_history[0]:.6f} → {self.loss_history[-1]:.6f}"
            )

            # Check convergence
            if len(self.loss_history) >= 100:
                recent_losses = self.loss_history[-100:]
                loss_std = np.std(recent_losses)
                if loss_std < 1e-6:
                    print(
                        "Training appears to have converged (loss standard deviation < 1e-6)"
                    )
                else:
                    print(
                        f"Training loss standard deviation over last 100 epochs: {loss_std:.8f}"
                    )

    def align_patches_iteratively(
        self,
        patch_graph: TGraph,
        num_iterations: int = 2,
        num_epochs: int = 1000,
        learning_rate: float = 0.01,
        device: str = "cpu",
        use_scale: bool | None = None,
        use_randomized_init: bool | None = None,
        randomized_method: str | None = None,
        orthogonal_reg_weight: float | None = None,
        batch_size: int | None = None,
        use_bfs_training: bool | None = None,
    ) -> "GeoAlignmentProblem":
        """
        Perform iterative alignment by running the alignment procedure multiple times.
        Each iteration uses the result of the previous iteration as starting point.

        Args:
            patch_graph: Pre-computed patch graph with patches as node features and overlap_nodes attribute
                        containing overlap information. The overlap_nodes must be a dict mapping
                        patch pairs (i,j) to lists of overlapping node indices.
            num_iterations: Number of alignment iterations to perform
            num_epochs: Number of training epochs per iteration
            learning_rate: Learning rate for optimization
            device: Device for computation
            use_scale: Whether to scale patches before applying transformations (overrides init setting)
            use_randomized_init: Whether to use randomized initialization (overrides init setting)
            randomized_method: Randomized method to use (overrides init setting)
            orthogonal_reg_weight: Weight for orthogonal regularization (overrides init setting)
            batch_size: Batch size for training (overrides init setting, 0 = full batch)
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
            "verbose": self.verbose,
            "min_overlap": self.min_overlap,
            "orthogonal_reg_weight": self.orthogonal_reg_weight,
            "batch_size": self.batch_size,
            "use_bfs_training": self.use_bfs_training,
        }

        if use_scale is not None:
            self.use_scale = use_scale

        # Extract patches from patch graph for iterative processing
        current_patches = [self._copy_patch(patch) for patch in patch_graph.patches]

        for iteration in range(num_iterations):
            if self.verbose:
                print(f"\n=== Alignment Iteration {iteration + 1}/{num_iterations} ===")

            # Create a new patch graph for this iteration with updated patches
            # For simplicity, we'll use the same connectivity structure but with updated patches
            current_patch_graph = patch_graph
            current_patch_graph.patches = current_patches

            # Run one full alignment iteration
            self.align_patches(
                patch_graph=current_patch_graph,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                device=device,
                use_randomized_init=use_randomized_init,
                randomized_method=randomized_method,
            )

            if iteration < num_iterations - 1:  # Don't reset after the last iteration
                # Get the transformed patches for next iteration
                current_patches = [self._copy_patch(patch) for patch in self.patches]

                if self.verbose:
                    print(
                        f"Iteration {iteration + 1} completed. Preparing for next iteration..."
                    )
                    # Show some statistics about current alignment quality
                    if hasattr(self, "loss_history") and self.loss_history:
                        print(f"  Final loss: {self.loss_history[-1]:.6f}")

                # Reset the aligner state for next iteration
                self._reset_for_next_iteration(original_params)

        if self.verbose:
            print(f"\nIterative alignment completed after {num_iterations} iterations")
            if hasattr(self, "loss_history") and self.loss_history:
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
        self.patch_overlap = {}
        self.patch_index = []
        self.patch_degrees = []
        self.scales = []
        self.shifts = []
        self.rotations = []
        self.device = "cpu"

        # Reset transformations if they exist
        if hasattr(self, "rotations"):
            self.rotations = []
        if hasattr(self, "shifts"):
            self.shifts = []
        if hasattr(self, "_aligned_embedding"):
            self._aligned_embedding = None

        # Restore original parameters
        self.verbose = original_params["verbose"]
        self.min_overlap = original_params["min_overlap"]
        self.orthogonal_reg_weight = original_params["orthogonal_reg_weight"]
        self.batch_size = original_params["batch_size"]
        self.use_bfs_training = original_params["use_bfs_training"]

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
            coordinates=patch.coordinates.copy()
            if patch.coordinates is not None
            else None,
        )
        return new_patch

    def _extract_transformations(self):
        """Extract weight matrices and biases from the trained model."""
        self.rotations = []
        self.shifts = []

        for i in range(self.n_patches):
            # Extract weight (rotation)
            layer = self.model.transformations[i]

            # For parametric models, extract the actual orthogonal matrix
            if (
                hasattr(layer, "parametrizations")
                and "weight" in layer.parametrizations
            ):
                # This is a parametrized layer - get the actual orthogonal matrix
                weight = layer.weight.detach().cpu().numpy()
            else:
                # Non-parametrized layer - use weight directly
                weight = layer.weight.detach().cpu().numpy()

            self.rotations.append(weight)

            # For translations: always use L2G-style synchronized translations
            # Will be computed using calc_synchronised_translations
            self.shifts.append(np.zeros(self.dim))

    def _apply_transformations(self):
        """Apply learned transformations to patch coordinates."""
        # First apply rotations
        for i, patch in enumerate(self.patches):
            # Apply rotation (orthogonal transformation from model)
            patch.coordinates = patch.coordinates @ self.rotations[i].T

        # Then compute and apply translations (always use L2G-style)
        # Use L2G-style synchronized translations
        translations = self.calc_synchronised_translations()
        for i, t in enumerate(translations):
            self.patches[i].coordinates += t
            self.shifts[i] = t  # Store computed translations
