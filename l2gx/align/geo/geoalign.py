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
    
    for layer in model.transformation:
        if hasattr(layer, 'weight') and layer.weight.requires_grad:
            W = layer.weight  # Shape: [dim, dim]
            
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
        orthogonal_reg_weight: float = 0.1,
        center_patches: bool = True,
        preserve_scale: bool = True,
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
        model = (
            AffineModel(dim, self.n_patches, device).to(device)
            if model_type == "affine"
            else OrthogonalModel(dim, self.n_patches, device).to(device)
        )
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

        res, loss_hist = self.train_alignment_model(
            embeddings,
            device=self.device,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            model_type=self.model_type,
            verbose=self.verbose,
        )

        self.loss_hist = loss_hist

        self.rotations = [
            res.transformation[i].weight.to("cpu").detach().numpy()
            for i in range(self.n_patches)
        ]
        self.shifts = [
            res.transformation[i].bias.to("cpu").detach().numpy()
            for i in range(self.n_patches)
        ]
        for i, patch in enumerate(self.patches):
            self.patches[i].coordinates = patch.coordinates @ self.rotations[i].T
            self.patches[i].coordinates += self.shifts[i]
        self._aligned_embedding = self.mean_embedding()