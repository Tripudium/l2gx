"""
Example demonstrating orthogonal regularization in the geometric aligner.

This shows how to use the new orthogonal regularization feature to encourage
weight matrices to be close to orthogonal transformations.
"""

import torch
import numpy as np
from l2gx.align import GeoAlignmentProblem
from l2gx.patch import Patch

def create_test_patches():
    """Create some test patches with known orthogonal transformations."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Original coordinates
    coords = np.random.randn(50, 2)
    
    # Create patches with orthogonal transformations
    patches = []
    
    # Patch 0: Identity (reference)
    patches.append(Patch(range(50), coords.copy()))
    
    # Patch 1: 45-degree rotation
    theta = np.pi / 4
    R1 = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
    coords1 = coords @ R1.T + np.array([1.0, 0.5])
    patches.append(Patch(range(50), coords1))
    
    # Patch 2: 90-degree rotation + translation
    theta = np.pi / 2
    R2 = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
    coords2 = coords @ R2.T + np.array([-0.5, 1.0])
    patches.append(Patch(range(50), coords2))
    
    return patches

def test_orthogonal_regularization():
    """Test the orthogonal regularization feature."""
    patches = create_test_patches()
    
    print("=== Testing WITHOUT orthogonal regularization ===")
    aligner_normal = GeoAlignmentProblem(
        num_epochs=200,
        learning_rate=0.01,
        model_type="affine",
        use_orthogonal_regularization=False,
        verbose=True
    )
    
    aligner_normal.align_patches(patches)
    
    # Check how orthogonal the learned matrices are
    print("\nOrthogonality check (normal training):")
    metrics_normal = aligner_normal.check_orthogonality(aligner_normal.model)
    for patch_name, metrics in metrics_normal.items():
        print(f"{patch_name}: Frobenius error = {metrics['frobenius_error']:.4f}, "
              f"Det = {metrics['determinant']:.4f}")
    
    print("\n" + "="*60)
    print("=== Testing WITH orthogonal regularization ===")
    aligner_reg = GeoAlignmentProblem(
        num_epochs=200,
        learning_rate=0.01,
        model_type="affine",
        use_orthogonal_regularization=True,
        orthogonal_reg_weight=0.1,
        verbose=True
    )
    
    aligner_reg.align_patches(patches)
    
    # Check how orthogonal the learned matrices are
    print("\nOrthogonality check (with regularization):")
    metrics_reg = aligner_reg.check_orthogonality(aligner_reg.model)
    for patch_name, metrics in metrics_reg.items():
        print(f"{patch_name}: Frobenius error = {metrics['frobenius_error']:.4f}, "
              f"Det = {metrics['determinant']:.4f}")
    
    print("\n=== Comparison ===")
    print("Without regularization:")
    for patch_name in metrics_normal:
        if patch_name != 'patch_0':  # Skip fixed patch
            print(f"  {patch_name}: Frobenius error = {metrics_normal[patch_name]['frobenius_error']:.4f}")
    
    print("With regularization:")
    for patch_name in metrics_reg:
        if patch_name != 'patch_0':  # Skip fixed patch
            print(f"  {patch_name}: Frobenius error = {metrics_reg[patch_name]['frobenius_error']:.4f}")

if __name__ == "__main__":
    test_orthogonal_regularization()