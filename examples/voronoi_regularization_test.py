"""
Test orthogonal regularization with real Voronoi patch data.

This test uses the same data generation as the demo but with noise=0 and no scaling
to verify that patches generated with only rotations and translations can be 
perfectly aligned using orthogonal transformations.
"""

import copy
import torch
import numpy as np
from l2gx.align import GeoAlignmentProblem
from l2gx.patch import Patch
from examples.example import generate_points, voronoi_patches, transform_patches

# Random number generator
rg = np.random.default_rng(42)

def test_voronoi_orthogonal_regularization():
    """Test orthogonal regularization on real Voronoi patches."""
    print("=== TESTING ORTHOGONAL REGULARIZATION WITH VORONOI PATCHES ===")
    print("Data: 5 clusters, no noise (std=0), no scaling")
    print("Patches: Voronoi tessellation with min_overlap=64")
    print("Transformations: Random rotations + translations (shift_scale=10)")
    
    # Generate points exactly as in demo but with no noise and no scaling
    np.random.seed(42)
    torch.manual_seed(42)
    
    points = generate_points(
        n_clusters=5, 
        scale=1.0,     # No scaling
        std=0.0,       # No noise  
        max_size=2000, 
        min_size=128, 
        dim=2
    )
    
    patches, centers = voronoi_patches(
        points, 
        sample_size=10, 
        min_degree=4, 
        min_overlap=64, 
        min_patch_size=128, 
        eps=1, 
        kmeans=False
    )
    
    print(f"\nGenerated {len(patches)} patches")
    print(f"Patch sizes: {[len(p.nodes) for p in patches]}")
    
    # Apply random transformations exactly as in the demo
    print("\nApplying random rotations and translations...")
    noise_level = 0  # No noise
    shift_scale = 10
    scale_range = None  # No scaling
    
    # Create transformed copies of the patches
    transformed_patches = [copy.deepcopy(p) for p in patches]
    
    # Add noise to the transformed patches (disabled in this test)
    if noise_level > 0:
        for patch in transformed_patches:
            noise = rg.normal(loc=0, scale=noise_level, size=patch.coordinates.shape)
            patch.coordinates += noise
    
    # Apply random rotations and translations
    transformed_patches = transform_patches(
        transformed_patches, 
        shift_scale=shift_scale, 
        scale_range=scale_range
    )
    
    # Count patch connectivity after transformation
    overlap_count = 0
    for i in range(len(transformed_patches)):
        for j in range(i+1, len(transformed_patches)):
            overlap = len(set(transformed_patches[i].nodes) & set(transformed_patches[j].nodes))
            if overlap >= 64:
                overlap_count += 1
    
    print(f"Patch pairs with sufficient overlap (≥64) after transformation: {overlap_count}")
    print("Note: Patches have been randomly rotated and translated")
    
    # Test WITHOUT regularization
    print("\n" + "="*60)
    print("=== WITHOUT ORTHOGONAL REGULARIZATION ===")
    
    aligner_normal = GeoAlignmentProblem(
        num_epochs=500,  # More epochs for convergence
        learning_rate=0.01,
        model_type="affine",
        use_orthogonal_reg=False,
        center_patches=True,
        verbose=True,
        min_overlap=64
    )
    
    # Make copies for fair comparison (use transformed patches)
    patches_normal = [Patch(p.nodes.copy(), p.coordinates.copy()) for p in transformed_patches]
    aligner_normal.align_patches(patches_normal, scale=False)
    
    print("\nOrthogonality check (normal training):")
    metrics_normal = aligner_normal.check_orthogonality(aligner_normal.model)
    
    normal_errors = []
    for patch_name, metrics in metrics_normal.items():
        if patch_name != 'patch_0':  # Skip reference patch
            error = metrics['frobenius_error']
            det = metrics['determinant']
            normal_errors.append(error)
            print(f"  {patch_name}: Frobenius error = {error:.4f}, Det = {det:.4f}")
    
    # Test WITH regularization (corrected parameters)
    print("\n" + "="*60)
    print("=== WITH ORTHOGONAL REGULARIZATION ===")
    
    aligner_reg = GeoAlignmentProblem(
        num_epochs=1000,             # More epochs for convergence with large translations
        learning_rate=0.01,
        model_type="affine",
        use_orthogonal_reg=True,
        orthogonal_reg_weight=10.0,  # Higher regularization weight for large translations
        center_patches=True,         # Crucial for large translations
        preserve_scale=False,        # Strict orthogonality
        verbose=True,
        min_overlap=64
    )
    
    # Make copies for fair comparison (use transformed patches)  
    patches_reg = [Patch(p.nodes.copy(), p.coordinates.copy()) for p in transformed_patches]
    aligner_reg.align_patches(patches_reg, scale=False)
    
    print("\nOrthogonality check (with regularization):")
    metrics_reg = aligner_reg.check_orthogonality(aligner_reg.model)
    
    reg_errors = []
    for patch_name, metrics in metrics_reg.items():
        if patch_name != 'patch_0':  # Skip reference patch
            error = metrics['frobenius_error']
            det = metrics['determinant']
            reg_errors.append(error)
            print(f"  {patch_name}: Frobenius error = {error:.4f}, Det = {det:.4f}")
    
    # Compare results
    print("\n" + "="*60)
    print("=== COMPARISON RESULTS ===")
    
    avg_error_normal = np.mean(normal_errors)
    avg_error_reg = np.mean(reg_errors)
    max_error_reg = np.max(reg_errors)
    
    print(f"Without regularization:")
    print(f"  Average Frobenius error: {avg_error_normal:.4f}")
    print(f"With regularization:")
    print(f"  Average Frobenius error: {avg_error_reg:.4f}")
    print(f"  Maximum Frobenius error: {max_error_reg:.4f}")
    
    improvement = (avg_error_normal - avg_error_reg) / avg_error_normal * 100
    print(f"\nImprovement: {improvement:.1f}%")
    
    # Success criteria
    print("\n" + "="*60)
    print("=== SUCCESS CRITERIA ===")
    
    excellent_threshold = 0.05
    good_threshold = 0.1
    
    excellent_patches = sum(1 for error in reg_errors if error < excellent_threshold)
    good_patches = sum(1 for error in reg_errors if error < good_threshold)
    
    print(f"Patches with excellent orthogonality (<{excellent_threshold}): {excellent_patches}/{len(reg_errors)}")
    print(f"Patches with good orthogonality (<{good_threshold}): {good_patches}/{len(reg_errors)}")
    
    if max_error_reg < excellent_threshold:
        print("✅ EXCELLENT: All patches achieve excellent orthogonality!")
    elif max_error_reg < good_threshold:
        print("✅ GOOD: All patches achieve good orthogonality!")
    elif good_patches >= len(reg_errors) * 0.8:
        print("⚠️  ACCEPTABLE: Most patches achieve good orthogonality")
    else:
        print("❌ POOR: Many patches fail to achieve good orthogonality")
    
    print(f"\nConclusion: Orthogonal regularization {'works excellently' if max_error_reg < excellent_threshold else 'works well' if max_error_reg < good_threshold else 'needs improvement'} on Voronoi patch data")
    
    return {
        'original_patches': patches,
        'transformed_patches': transformed_patches, 
        'normal_metrics': metrics_normal,
        'reg_metrics': metrics_reg,
        'improvement': improvement,
        'max_error': max_error_reg
    }

if __name__ == "__main__":
    results = test_voronoi_orthogonal_regularization()