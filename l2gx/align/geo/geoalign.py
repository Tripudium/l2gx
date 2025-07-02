"""
Alignment using pytorch geometric.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from l2gx.align.registry import register_aligner
from l2gx.align.alignment import AlignmentProblem
from l2gx.patch import Patch
from l2gx.align.geo.model import AffineModel, OrthogonalModel
from l2gx.align.utils import to_device

def patchgraph_mse_loss(transformed_emb):
    """
    Custom loss function that computes the squared norm of differences
    between transformed pairs in the dictionary.

    Args:
        transformed_dict: Dictionary with keys (i,j) and values (XW_i+b_i, YW_j+b_j)

    Returns:
        Total loss as the sum of squared differences
    """
    total_loss = 0.0

    for (_, _), (transformed_X, transformed_Y) in transformed_emb.items():
        # Calculate squared norm of the difference
        pair_loss = F.mse_loss(transformed_X, transformed_Y, reduction="mean")
        total_loss += pair_loss

    return total_loss


def orthogonal_regularization_loss(model, regularization_weight=1.0, preserve_scale=True):
    """
    Compute orthogonal regularization loss for weight matrices.
    
    For each weight matrix W, computes ||W @ W.T - I||_F^2
    This encourages the weight matrices to be close to orthogonal.
    
    Args:
        model: The model containing weight matrices to regularize
        regularization_weight: Strength of the regularization term
        preserve_scale: If True, allows uniform scaling (W @ W.T = s * I for some s > 0)
        
    Returns:
        Regularization loss scalar
    """
    reg_loss = 0.0
    count = 0
    
    # Handle both old and new model structures
    if hasattr(model, 'weights'):
        # New structure with separate weights and translations
        weight_list = model.weights
    else:
        # Old structure with linear layers
        weight_list = [layer.weight for layer in model.transformation if hasattr(layer, 'weight')]
    
    for W in weight_list:
        if W.requires_grad:
            
            # Compute W @ W.T
            WWT = torch.mm(W, W.t())
            
            if preserve_scale:
                # Allow uniform scaling: minimize ||W @ W.T - s * I||_F^2 where s is optimal
                # Optimal s = trace(W @ W.T) / dim
                dim = W.shape[0]
                s = torch.trace(WWT) / dim
                target = s * torch.eye(dim, device=W.device, dtype=W.dtype)
            else:
                # Strict orthogonality: W @ W.T = I
                target = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
            
            orthogonal_deviation = WWT - target
            
            # Squared Frobenius norm
            reg_loss += torch.sum(orthogonal_deviation ** 2)
            count += 1
    
    # Average over all regularized layers
    if count > 0:
        reg_loss = reg_loss / count
    
    return regularization_weight * reg_loss


def patchgraph_loss_with_reg(transformed_emb, model, orthogonal_reg_weight=0.1, preserve_scale=True):
    """
    Combined loss function with MSE alignment loss and orthogonal regularization.
    
    Args:
        transformed_emb: Dictionary of transformed embeddings
        model: The model containing weight matrices
        orthogonal_reg_weight: Weight for orthogonal regularization term
        preserve_scale: If True, allows uniform scaling in regularization
        
    Returns:
        Total loss (MSE + regularization)
    """
    # Main alignment loss
    mse_loss = patchgraph_mse_loss(transformed_emb)
    
    # Orthogonal regularization loss
    reg_loss = orthogonal_regularization_loss(model, orthogonal_reg_weight, preserve_scale)
    
    return mse_loss + reg_loss

@register_aligner("geo")
class GeoAlignmentProblem(AlignmentProblem):
    """
    Alignment problem using pytorch geometric.
    """

    def __init__(
        self,
        num_epochs: int = 1000,
        learning_rate: float = 0.001,
        model_type: str = "affine",
        device: str = "cpu",
        use_orthogonal_reg: bool = True,
        orthogonal_reg_weight: float = 10.0,
        center_patches: bool = True,
        preserve_scale: bool = False,
        use_two_stage_training: bool = False,
        stage1_epochs: int = None,
        stage2_epochs: int = None,
        verbose: bool = False,
        min_overlap: int | None = None
    ):
        super().__init__(
            verbose=verbose,
            min_overlap=min_overlap
        )
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.model_type = model_type
        self.use_orthogonal_reg = use_orthogonal_reg
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.center_patches = center_patches
        self.preserve_scale = preserve_scale
        self.use_two_stage_training = use_two_stage_training
        self.stage1_epochs = stage1_epochs or (num_epochs * 2 // 3)  # 2/3 for rotations
        self.stage2_epochs = stage2_epochs or (num_epochs // 3)      # 1/3 for translations
        self.loss_hist = []

    def set_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate

    def set_num_epochs(self, num_epochs: int):
        self.num_epochs = num_epochs

    def _get_intersections(self, center_patches=True):
        """
        Calculate the intersection of nodes between patches.
        
        Args:
            center_patches: If True, center each patch to have zero mean before computing intersections
        """
        intersections = {}
        embeddings = {}
        for i, _ in enumerate(self.patches):
            for j in range(len(self.patches)):
                if i < j:
                    intersections[(i, j)] = list(
                        set(self.patches[i].nodes.tolist()).intersection(
                            set(self.patches[j].nodes.tolist())
                        )
                    )
                    if len(intersections[(i, j)]) >= self.min_overlap:
                        coords_i = self.patches[i].get_coordinates(list(intersections[(i, j)]))
                        coords_j = self.patches[j].get_coordinates(list(intersections[(i, j)]))
                        
                        # Center the coordinates if requested (important for orthogonal regularization)
                        if center_patches:
                            coords_i = coords_i - np.mean(coords_i, axis=0)
                            coords_j = coords_j - np.mean(coords_j, axis=0)
                        
                        embeddings[(i, j)] = [
                            torch.tensor(coords_i),
                            torch.tensor(coords_j),
                        ]
        # embeddings = list(itertools.chain.from_iterable(embeddings))
        return intersections, embeddings
    
    def train_two_stage_alignment(self, embeddings, device="cpu", verbose=True):
        """
        Two-stage training: first learn rotations on centered data, then learn translations.
        
        Stage 1: Learn orthogonal transformations W on centered coordinates
        Stage 2: Fix W and learn translations b on non-centered coordinates
        """
        patch_emb = to_device(embeddings, device)
        dim = patch_emb[list(patch_emb.keys())[0]][0].shape[1]
        
        if verbose:
            print("=== TWO-STAGE TRAINING ===")
            print(f"Stage 1: Learn rotations ({self.stage1_epochs} epochs)")
            print(f"Stage 2: Learn translations ({self.stage2_epochs} epochs)")
        
        # === STAGE 1: Learn Rotations on Centered Data ===
        if verbose:
            print("\nSTAGE 1: Learning rotations on centered coordinates...")
        
        # Create centered embeddings
        centered_embeddings = {}
        for (i, j), (X, Y) in patch_emb.items():
            # Center each coordinate set
            X_centered = X - torch.mean(X, dim=0, keepdim=True)
            Y_centered = Y - torch.mean(Y, dim=0, keepdim=True)
            centered_embeddings[(i, j)] = (X_centered, Y_centered)
        
        # Create rotation-only model (no bias terms)
        if self.model_type == "affine":
            model = AffineModel(dim, self.n_patches, device).to(device)
            # Disable bias terms for stage 1
            for layer in model.transformation:
                layer.bias.requires_grad = False
                layer.bias.data.zero_()
        else:
            model = OrthogonalModel(dim, self.n_patches, device).to(device)
        
        # Stage 1 training: rotation only with orthogonal regularization
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=self.learning_rate)
        stage1_losses = []
        
        for epoch in range(self.stage1_epochs):
            optimizer.zero_grad()
            transformed_emb = model(centered_embeddings)
            
            # Use orthogonal regularization for rotations
            mse_loss = patchgraph_mse_loss(transformed_emb)
            if self.use_orthogonal_reg:
                reg_loss = orthogonal_regularization_loss(model, self.orthogonal_reg_weight, self.preserve_scale)
                total_loss = mse_loss + reg_loss
            else:
                total_loss = mse_loss
            
            total_loss.backward()
            optimizer.step()
            stage1_losses.append(total_loss.item())
            
            if verbose and epoch % max(1, self.stage1_epochs // 10) == 0:
                print(f"  Epoch {epoch}: loss={total_loss.item():.6f}")
        
        if verbose:
            print(f"Stage 1 complete. Final rotation loss: {stage1_losses[-1]:.6f}")
        
        # === STAGE 2: Learn Translations on Original Data ===
        if verbose:
            print("\nSTAGE 2: Learning translations on original coordinates...")
        
        # Freeze rotation parameters and enable bias terms
        if self.model_type == "affine":
            for layer in model.transformation:
                layer.weight.requires_grad = False  # Freeze rotations
                layer.bias.requires_grad = True     # Enable translations
        
        # Stage 2 training: translation only, no regularization
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], 
                              lr=self.learning_rate * 0.1)  # Lower LR for translations
        stage2_losses = []
        
        for epoch in range(self.stage2_epochs):
            optimizer.zero_grad()
            transformed_emb = model(patch_emb)  # Use original (non-centered) data
            
            # Only MSE loss for translation learning
            loss = patchgraph_mse_loss(transformed_emb)
            loss.backward()
            optimizer.step()
            stage2_losses.append(loss.item())
            
            if verbose and epoch % max(1, self.stage2_epochs // 5) == 0:
                print(f"  Epoch {epoch}: loss={loss.item():.6f}")
        
        if verbose:
            print(f"Stage 2 complete. Final translation loss: {stage2_losses[-1]:.6f}")
        
        # Combine loss histories
        combined_losses = stage1_losses + stage2_losses
        
        return model, combined_losses
    
    def train_alignment_model(
        self,
        embeddings,
        device="cpu",
        num_epochs=100,
        learning_rate=0.05,
        model_type="affine",
        verbose=True,
    ):
        """
        Train the model on the patch embeddings
        Args:
            patch_emb: list of torch.Tensor
                patch embeddings
            n_patches: int
                number of patches
            device: str
                device to run the model on
            num_epochs: int
                number of epochs to train the model
            learning_rate: float
                learning rate for the optimizer
        Returns:
            model: Model
            loss_hist: list
        """
        patch_emb = to_device(embeddings, device)
        dim = patch_emb[list(patch_emb.keys())[0]][0].shape[1]
        if model_type == "affine":
            model = AffineModel(dim, self.n_patches, device).to(device)
        else:
            model = OrthogonalModel(dim, self.n_patches, device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_hist = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            transformed_patch_emb = model(patch_emb)
            
            # Choose loss function based on regularization setting
            if self.use_orthogonal_reg:
                loss = patchgraph_loss_with_reg(
                    transformed_patch_emb, model, self.orthogonal_reg_weight, self.preserve_scale
                )
            else:
                loss = patchgraph_mse_loss(transformed_patch_emb)
                
            loss.backward(retain_graph=True)
            optimizer.step()
            loss_hist.append(loss.item())
            
            if verbose:
                if epoch % 10 == 0:
                    if self.use_orthogonal_reg:
                        # Also report the individual loss components
                        with torch.no_grad():
                            mse_loss = patchgraph_mse_loss(transformed_patch_emb)
                            reg_loss = orthogonal_regularization_loss(model, self.orthogonal_reg_weight, self.preserve_scale)
                        print(f"Epoch {epoch}, Total Loss: {loss.item():.6f}, "
                              f"MSE: {mse_loss.item():.6f}, Reg: {reg_loss.item():.6f}")
                    else:
                        print(f"Epoch {epoch}, Loss: {loss.item()}")

        return model, loss_hist
    
    def check_orthogonality(self, model):
        """
        Check how close the learned weight matrices are to being orthogonal.
        
        Args:
            model: Trained model
            
        Returns:
            Dictionary with orthogonality metrics for each transformation
        """
        orthogonality_metrics = {}
        
        for i, layer in enumerate(model.transformation):
            if hasattr(layer, 'weight'):
                W = layer.weight.detach()
                
                # Compute W @ W.T
                WWT = torch.mm(W, W.t())
                Id = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
                
                # Orthogonality deviation
                deviation = WWT - Id
                frobenius_error = torch.norm(deviation, 'fro').item()
                
                # Determinant (should be ±1 for orthogonal matrices)
                det = torch.det(W).item()
                
                # Singular values (should be close to 1 for orthogonal matrices)
                U, s, Vh = torch.linalg.svd(W)
                singular_values = s.detach().cpu().numpy()
                
                orthogonality_metrics[f'patch_{i}'] = {
                    'frobenius_error': frobenius_error,
                    'determinant': det,
                    'singular_values': singular_values,
                    'condition_number': (s.max() / s.min()).item()
                }
        
        return orthogonality_metrics
    
    def debug_regularization(self, model, sample_embeddings):
        """
        Debug the orthogonal regularization to see what's going wrong.
        """
        print("=== REGULARIZATION DEBUG ===")
        
        # Check model structure
        print(f"Model type: {type(model).__name__}")
        print(f"Has weights attr: {hasattr(model, 'weights')}")
        print(f"Has transformation attr: {hasattr(model, 'transformation')}")
        
        # Check what the regularization function sees
        if hasattr(model, 'weights'):
            weight_list = model.weights
            print(f"Using model.weights: {len(weight_list)} weights")
        else:
            weight_list = [layer.weight for layer in model.transformation if hasattr(layer, 'weight')]
            print(f"Using model.transformation: {len(weight_list)} weights")
        
        # Check each weight matrix
        total_reg_loss = 0
        for i, W in enumerate(weight_list):
            if W.requires_grad:
                print(f"\nWeight matrix {i}:")
                print(f"  Shape: {W.shape}")
                print(f"  Requires grad: {W.requires_grad}")
                
                # Compute orthogonality metrics
                with torch.no_grad():
                    W_np = W.cpu().numpy()
                    WWT = W_np @ W_np.T
                    I = np.eye(W.shape[0])
                    
                    # Frobenius error
                    if self.preserve_scale:
                        s = np.trace(WWT) / W.shape[0]
                        target = s * I
                        print(f"  Scale factor s: {s:.3f}")
                    else:
                        target = I
                    
                    deviation = WWT - target
                    frobenius_error = np.linalg.norm(deviation, 'fro')
                    det = np.linalg.det(W_np)
                    
                    print(f"  Determinant: {det:.3f}")
                    print(f"  Frobenius error: {frobenius_error:.3f}")
                    print(f"  WWT diagonal: {np.diag(WWT)}")
                    print(f"  WWT off-diagonal max: {np.max(np.abs(WWT - np.diag(np.diag(WWT)))):.3f}")
                
                # Compute regularization loss contribution
                WWT_torch = torch.mm(W, W.t())
                if self.preserve_scale:
                    dim = W.shape[0]
                    s = torch.trace(WWT_torch) / dim
                    target_torch = s * torch.eye(dim, device=W.device, dtype=W.dtype)
                else:
                    target_torch = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
                
                deviation_torch = WWT_torch - target_torch
                reg_contribution = torch.sum(deviation_torch ** 2).item()
                total_reg_loss += reg_contribution
                
                print(f"  Regularization contribution: {reg_contribution:.6f}")
        
        print(f"\nTotal regularization loss: {total_reg_loss:.6f}")
        print(f"Weighted regularization loss: {total_reg_loss * self.orthogonal_reg_weight:.6f}")
        
        # Compare with MSE loss
        if sample_embeddings:
            with torch.no_grad():
                transformed_emb = model(sample_embeddings)
                mse_loss = patchgraph_mse_loss(transformed_emb).item()
                print(f"MSE loss: {mse_loss:.6f}")
                print(f"Regularization/MSE ratio: {(total_reg_loss * self.orthogonal_reg_weight) / mse_loss:.3f}")
                
                if (total_reg_loss * self.orthogonal_reg_weight) / mse_loss < 0.01:
                    print("⚠️  WARNING: Regularization is much smaller than MSE loss!")
                    print("   Consider increasing orthogonal_reg_weight")
        
        return {
            'total_reg_loss': total_reg_loss,
            'weighted_reg_loss': total_reg_loss * self.orthogonal_reg_weight,
            'num_weights': len([W for W in weight_list if W.requires_grad])
        }
    
    def debug_alignment_step_by_step(self, patches, ground_truth_patches=None):
        """
        Debug the alignment process step by step to understand what's going wrong.
        """
        print("=== STEP-BY-STEP ALIGNMENT DEBUG ===")
        
        # Step 1: Check initial patch positions
        print("\n1. INITIAL PATCH POSITIONS:")
        for i, patch in enumerate(patches):
            center = np.mean(patch.coordinates, axis=0)
            std = np.std(patch.coordinates, axis=0)
            print(f"   Patch {i}: center=({center[0]:.2f}, {center[1]:.2f}), std=({std[0]:.2f}, {std[1]:.2f})")
        
        # Step 2: Check intersections before alignment
        print("\n2. CHECKING INTERSECTIONS:")
        self._register_patches(patches)
        intersections, embeddings = self._get_intersections(center_patches=self.center_patches)
        
        for (i, j), nodes in intersections.items():
            if len(nodes) >= self.min_overlap:
                coords_i = patches[i].get_coordinates(nodes)
                coords_j = patches[j].get_coordinates(nodes)
                
                # Compute the "ground truth" transformation between these overlaps
                if len(nodes) >= 3:
                    # Center both sets
                    center_i = np.mean(coords_i, axis=0)
                    center_j = np.mean(coords_j, axis=0)
                    centered_i = coords_i - center_i
                    centered_j = coords_j - center_j
                    
                    # Find optimal rotation using Procrustes
                    if np.linalg.norm(centered_i) > 1e-6 and np.linalg.norm(centered_j) > 1e-6:
                        H = centered_i.T @ centered_j
                        U, s, Vt = np.linalg.svd(H)
                        R_optimal = Vt.T @ U.T
                        if np.linalg.det(R_optimal) < 0:
                            Vt[-1, :] *= -1
                            R_optimal = Vt.T @ U.T
                        
                        # Compute alignment error with optimal transformation
                        aligned_i = centered_i @ R_optimal.T + center_j
                        optimal_error = np.mean(np.sum((aligned_i - coords_j)**2, axis=1))
                        
                        print(f"   Patches {i}-{j}: {len(nodes)} nodes, optimal_error={optimal_error:.4f}")
                        print(f"      Translation: ({center_j[0]-center_i[0]:.3f}, {center_j[1]-center_i[1]:.3f})")
                        print(f"      Rotation det: {np.linalg.det(R_optimal):.3f}")
        
        # Step 3: Train and check what was learned
        print(f"\n3. TRAINING MODEL (centering={self.center_patches}):")
        
        res, loss_hist = self.train_alignment_model(
            embeddings,
            device=self.device,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            model_type=self.model_type,
            verbose=False,
        )
        
        print(f"   Final loss: {loss_hist[-1]:.6f}")
        print(f"   Loss reduction: {(loss_hist[0]-loss_hist[-1])/loss_hist[0]*100:.1f}%")
        
        # Step 4: Check what transformations were learned
        print("\n4. LEARNED TRANSFORMATIONS:")
        if hasattr(res, 'weights'):
            weights = [w.detach().cpu().numpy() for w in res.weights]
            translations = [t.detach().cpu().numpy() for t in res.translations]
        else:
            weights = [layer.weight.detach().cpu().numpy() for layer in res.transformation]
            translations = [layer.bias.detach().cpu().numpy() for layer in res.transformation]
        
        for i in range(self.n_patches):
            W = weights[i]
            b = translations[i]
            det = np.linalg.det(W)
            trans_mag = np.linalg.norm(b)
            
            print(f"   Patch {i}: det(W)={det:.3f}, |translation|={trans_mag:.3f}")
            if i > 0:  # Skip reference patch
                print(f"      Translation: ({b[0]:.3f}, {b[1]:.3f})")
        
        # Step 5: Apply transformations and check final alignment
        print("\n5. CHECKING FINAL ALIGNMENT:")
        
        # Apply transformations manually to see what happens
        transformed_patches = []
        for i, patch in enumerate(patches):
            coords = patch.coordinates.copy()
            
            if hasattr(res, 'use_rotated_translation') and res.use_rotated_translation:
                new_coords = (coords + translations[i]) @ weights[i].T
            else:
                new_coords = coords @ weights[i].T + translations[i]
            
            transformed_patches.append(Patch(patch.nodes.copy(), new_coords))
        
        # Check alignment quality on overlaps
        total_error = 0
        total_points = 0
        
        for (i, j), nodes in intersections.items():
            if len(nodes) >= self.min_overlap:
                coords_i = transformed_patches[i].get_coordinates(nodes)
                coords_j = transformed_patches[j].get_coordinates(nodes)
                
                error = np.mean(np.sum((coords_i - coords_j)**2, axis=1))
                total_error += error * len(nodes)
                total_points += len(nodes)
                
                print(f"   Patches {i}-{j}: alignment_error={np.sqrt(error):.4f}")
        
        if total_points > 0:
            avg_error = np.sqrt(total_error / total_points)
            print(f"   Average alignment RMSE: {avg_error:.4f}")
        
        # Step 6: Compare with ground truth if available
        if ground_truth_patches is not None:
            print("\n6. COMPARISON WITH GROUND TRUTH:")
            for i, (transformed, gt) in enumerate(zip(transformed_patches, ground_truth_patches)):
                if len(set(transformed.nodes) & set(gt.nodes)) > 0:
                    common_nodes = list(set(transformed.nodes) & set(gt.nodes))
                    trans_coords = transformed.get_coordinates(common_nodes)
                    gt_coords = gt.get_coordinates(common_nodes)
                    
                    # Center both for comparison
                    trans_center = np.mean(trans_coords, axis=0)
                    gt_center = np.mean(gt_coords, axis=0)
                    
                    center_diff = np.linalg.norm(trans_center - gt_center)
                    print(f"   Patch {i}: center_difference={center_diff:.3f}")
        
        return {
            'transformed_patches': transformed_patches,
            'learned_weights': weights,
            'learned_translations': translations,
            'final_loss': loss_hist[-1]
        }
    
    def check_overlap_statistics(self, verbose=True):
        """
        Check the quality and statistics of patch overlaps.
        
        Returns:
            Dictionary with overlap statistics and diagnostics
        """
        if not hasattr(self, 'intersections'):
            print("Error: No intersections found. Call align_patches() first.")
            return {}
        
        stats = {}
        
        # Basic overlap statistics
        total_overlaps = len(self.intersections)
        overlap_sizes = [len(nodes) for nodes in self.intersections.values()]
        
        stats['total_overlaps'] = total_overlaps
        stats['overlap_sizes'] = overlap_sizes
        stats['min_overlap'] = min(overlap_sizes) if overlap_sizes else 0
        stats['max_overlap'] = max(overlap_sizes) if overlap_sizes else 0
        stats['mean_overlap'] = sum(overlap_sizes) / len(overlap_sizes) if overlap_sizes else 0
        
        # Check if we have sufficient overlaps for alignment
        expected_overlaps = self.n_patches * (self.n_patches - 1) // 2
        stats['overlap_coverage'] = total_overlaps / expected_overlaps if expected_overlaps > 0 else 0
        
        # Check overlap quality for each patch pair
        alignment_quality = {}
        if hasattr(self, 'model'):
            for (i, j), nodes in self.intersections.items():
                if len(nodes) >= self.min_overlap:
                    # Get coordinates from both patches
                    coords_i = self.patches[i].get_coordinates(nodes)
                    coords_j = self.patches[j].get_coordinates(nodes)
                    
                    # Apply learned transformations
                    coords_i_tensor = torch.tensor(coords_i, dtype=torch.float32, device=self.device)
                    coords_j_tensor = torch.tensor(coords_j, dtype=torch.float32, device=self.device)
                    
                    with torch.no_grad():
                        transformed_i = self.model.transformation[i](coords_i_tensor)
                        transformed_j = self.model.transformation[j](coords_j_tensor)
                        
                        # Compute alignment error
                        alignment_error = torch.mean(torch.sum((transformed_i - transformed_j)**2, dim=1)).item()
                        alignment_quality[(i, j)] = {
                            'overlap_size': len(nodes),
                            'alignment_error': alignment_error,
                            'rmse': np.sqrt(alignment_error)
                        }
        
        stats['alignment_quality'] = alignment_quality
        
        if verbose:
            print("=== Overlap Statistics ===")
            print(f"Total overlaps found: {total_overlaps} (expected: {expected_overlaps})")
            print(f"Overlap coverage: {stats['overlap_coverage']:.2%}")
            print(f"Overlap sizes: min={stats['min_overlap']}, max={stats['max_overlap']}, mean={stats['mean_overlap']:.1f}")
            print(f"Minimum required overlap: {self.min_overlap}")
            
            if overlap_sizes and min(overlap_sizes) < self.min_overlap:
                print("⚠️  WARNING: Some overlaps are below minimum threshold!")
            
            print("\n=== Alignment Quality ===")
            if alignment_quality:
                for (i, j), quality in alignment_quality.items():
                    print(f"Patches {i}-{j}: {quality['overlap_size']} nodes, RMSE={quality['rmse']:.4f}")
                
                avg_rmse = np.mean([q['rmse'] for q in alignment_quality.values()])
                print(f"Average alignment RMSE: {avg_rmse:.4f}")
                
                if avg_rmse > 1.0:
                    print("⚠️  WARNING: High alignment errors detected!")
                    print("   This suggests patches are not aligning properly.")
                    print("   Consider: lower learning rate, more epochs, or different regularization.")
            else:
                print("No alignment quality data available (model not trained yet?)")
        
        return stats
    
    def diagnose_alignment_issues(self):
        """
        Comprehensive diagnosis of potential alignment issues.
        """
        print("=== Alignment Diagnosis ===")
        
        # Check basic setup
        print(f"Number of patches: {self.n_patches}")
        print(f"Minimum overlap requirement: {self.min_overlap}")
        print(f"Using orthogonal regularization: {self.use_orthogonal_reg}")
        if self.use_orthogonal_reg:
            print(f"Orthogonal regularization weight: {self.orthogonal_reg_weight}")
        print(f"Center patches: {self.center_patches}")
        
        # Check overlap statistics
        overlap_stats = self.check_overlap_statistics(verbose=False)
        
        # Identify potential issues
        issues = []
        suggestions = []
        
        if overlap_stats.get('overlap_coverage', 0) < 0.5:
            issues.append("Low overlap coverage between patches")
            suggestions.append("Increase patch overlap when creating patches")
        
        if overlap_stats.get('min_overlap', 0) < 3:
            issues.append("Very small overlaps detected")
            suggestions.append("Ensure patches have meaningful overlap (>= 3-5 points)")
        
        if hasattr(self, 'loss_hist') and len(self.loss_hist) > 10:
            final_loss = self.loss_hist[-1]
            initial_loss = self.loss_hist[0]
            if final_loss > 0.8 * initial_loss:
                issues.append("Poor loss convergence")
                suggestions.append("Try: lower learning rate, more epochs, or different initialization")
        
        alignment_quality = overlap_stats.get('alignment_quality', {})
        if alignment_quality:
            avg_rmse = np.mean([q['rmse'] for q in alignment_quality.values()])
            if avg_rmse > 1.0:
                issues.append(f"High alignment errors (avg RMSE: {avg_rmse:.3f})")
                suggestions.append("Try: reduce orthogonal regularization weight or disable it temporarily")
        
        # Check for centering issues
        if self.center_patches and alignment_quality:
            high_error_pairs = [(pair, q['rmse']) for pair, q in alignment_quality.items() if q['rmse'] > 2.0]
            if len(high_error_pairs) > len(alignment_quality) // 2:
                issues.append("Many patch pairs have high alignment errors")
                suggestions.append("Try setting center_patches=False - centering might interfere with alignment")
        
        # Report findings
        if issues:
            print("\n⚠️  ISSUES DETECTED:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
            
            print("\n💡 SUGGESTIONS:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")
        else:
            print("\n✅ No obvious issues detected in the alignment setup.")
            print("   If patches still don't align properly, the issue might be:")
            print("   - Insufficient training epochs")
            print("   - Learning rate too high/low")
            print("   - Complex patch transformations that are hard to learn")
        
        return {
            'issues': issues,
            'suggestions': suggestions,
            'stats': overlap_stats
        }

    def align_patches(self, patches: list[Patch], min_overlap: int | None = None, scale: bool = True):
        """
        Align the patches.
        """
        self._register_patches(patches, min_overlap)
        if scale:
            self.scale_patches()
        
        intersections, embeddings = self._get_intersections(center_patches=self.center_patches)
        self.intersections = intersections

        if self.use_two_stage_training:
            res, loss_hist = self.train_two_stage_alignment(
                embeddings,
                device=self.device,
                verbose=self.verbose,
            )
        else:
            res, loss_hist = self.train_alignment_model(
                embeddings,
                device=self.device,
                num_epochs=self.num_epochs,
                learning_rate=self.learning_rate,
                model_type=self.model_type,
                verbose=self.verbose,
            )

        self.loss_hist = loss_hist
        self.model = res  # Store the trained model for further analysis

        # Extract transformations (handle both old and new model structures)
        if hasattr(res, 'weights'):
            # New structure with separate weights and translations
            self.rotations = [
                res.weights[i].to("cpu").detach().numpy()
                for i in range(self.n_patches)
            ]
            self.shifts = [
                res.translations[i].to("cpu").detach().numpy()
                for i in range(self.n_patches)
            ]
        else:
            # Old structure with linear layers
            self.rotations = [
                res.transformation[i].weight.to("cpu").detach().numpy()
                for i in range(self.n_patches)
            ]
            self.shifts = [
                res.transformation[i].bias.to("cpu").detach().numpy()
                for i in range(self.n_patches)
            ]
        
        # Apply transformations to patches
        for i, patch in enumerate(self.patches):
            if hasattr(res, 'use_rotated_translation') and res.use_rotated_translation:
                # Use W @ (x + b) transformation
                self.patches[i].coordinates = (patch.coordinates + self.shifts[i]) @ self.rotations[i].T
            else:
                # Use standard W @ x + b transformation
                self.patches[i].coordinates = patch.coordinates @ self.rotations[i].T
                self.patches[i].coordinates += self.shifts[i]
        self._aligned_embedding = self.mean_embedding()