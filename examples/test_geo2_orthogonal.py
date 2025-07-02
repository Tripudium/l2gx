"""
Test orthogonal regularization in geo2 aligner.
"""

import copy
import numpy as np
import torch
from l2gx.align.geo2 import GeoAlignmentProblem2
from l2gx.patch import Patch
from examples.example import generate_points, voronoi_patches, transform_patches


def check_orthogonality(transformation_matrices):
    """Check how orthogonal the transformation matrices are."""
    orthogonality_errors = []
    
    for i, W in enumerate(transformation_matrices):
        if i == 0:  # Skip the fixed identity matrix
            continue
            
        # Compute W @ W.T
        WWT = W @ W.T
        
        # Compute ||W @ W.T - I||²_F
        I = np.eye(W.shape[0])
        diff = WWT - I
        frobenius_error = np.sum(diff * diff)
        orthogonality_errors.append(frobenius_error)
    
    return orthogonality_errors


def test_orthogonal_regularization():
    """Test the effect of orthogonal regularization."""
    print("=== TESTING ORTHOGONAL REGULARIZATION IN GEO2 ===")
    
    # Generate test data
    np.random.seed(42)
    torch.manual_seed(42)
    
    points = generate_points(n_clusters=2, scale=1.0, std=0.0, max_size=150, min_size=100, dim=2)
    patches, _ = voronoi_patches(points, sample_size=3, min_degree=4, min_overlap=30, min_patch_size=50, eps=1, kmeans=False)
    
    print(f"Generated {len(patches)} patches from {points.shape[0]} points")
    
    # Apply transformations
    np.random.seed(42)
    transformed_patches = transform_patches([copy.deepcopy(p) for p in patches], shift_scale=1.0, scale_range=None)
    
    # Test WITHOUT orthogonal regularization
    print("\n--- Testing WITHOUT orthogonal regularization ---")
    aligner_no_reg = GeoAlignmentProblem2(verbose=False, use_orthogonal_reg=False)
    
    result_no_reg = aligner_no_reg.align_patches(
        patches=[copy.deepcopy(p) for p in transformed_patches],
        num_epochs=300,
        learning_rate=0.01
    )
    
    # Check orthogonality
    orthogonality_no_reg = check_orthogonality(result_no_reg.rotations)
    print(f"Final loss: {result_no_reg.loss_history[-1]:.6f}")
    print(f"Orthogonality errors: {[f'{err:.6f}' for err in orthogonality_no_reg]}")
    print(f"Mean orthogonality error: {np.mean(orthogonality_no_reg):.6f}")
    
    # Test WITH orthogonal regularization
    print("\n--- Testing WITH orthogonal regularization ---")
    aligner_with_reg = GeoAlignmentProblem2(verbose=False, use_orthogonal_reg=True, orthogonal_reg_weight=10.0)
    
    result_with_reg = aligner_with_reg.align_patches(
        patches=[copy.deepcopy(p) for p in transformed_patches],
        num_epochs=300,
        learning_rate=0.01
    )
    
    # Check orthogonality
    orthogonality_with_reg = check_orthogonality(result_with_reg.rotations)
    print(f"Final loss: {result_with_reg.loss_history[-1]:.6f}")
    print(f"Orthogonality errors: {[f'{err:.6f}' for err in orthogonality_with_reg]}")
    print(f"Mean orthogonality error: {np.mean(orthogonality_with_reg):.6f}")
    
    # Compare results
    print(f"\n--- Comparison ---")
    improvement = (np.mean(orthogonality_no_reg) - np.mean(orthogonality_with_reg)) / np.mean(orthogonality_no_reg) * 100
    print(f"Orthogonality improvement with regularization: {improvement:.1f}%")
    
    if improvement > 50:
        print("✅ Orthogonal regularization significantly improves matrix orthogonality!")
    elif improvement > 10:
        print("✅ Orthogonal regularization improves matrix orthogonality")
    else:
        print("⚠️  Orthogonal regularization has limited effect")


if __name__ == "__main__":
    test_orthogonal_regularization()