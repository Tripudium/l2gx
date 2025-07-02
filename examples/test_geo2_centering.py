"""
Test the effect of patch centering in geo2 aligner.
"""

import copy
import numpy as np
import torch
from l2gx.align.geo2 import GeoAlignmentProblem2
from examples.example import generate_points, voronoi_patches, transform_patches


def test_centering_effect():
    """Compare alignment with and without patch centering."""
    print("=== TESTING PATCH CENTERING EFFECT ===")
    
    # Generate test data
    np.random.seed(42)
    torch.manual_seed(42)
    
    points = generate_points(n_clusters=3, scale=1.0, std=0.0, max_size=200, min_size=150, dim=2)
    patches, _ = voronoi_patches(points, sample_size=4, min_degree=4, min_overlap=50, min_patch_size=80, eps=1, kmeans=False)
    
    print(f"Generated {len(patches)} patches from {points.shape[0]} points")
    
    # Apply transformations
    np.random.seed(42)
    transformed_patches = transform_patches([copy.deepcopy(p) for p in patches], shift_scale=2.0, scale_range=None)
    
    # Test WITHOUT centering
    print("\n--- Testing WITHOUT patch centering ---")
    aligner_no_center = GeoAlignmentProblem2(
        verbose=False, 
        use_orthogonal_reg=True, 
        orthogonal_reg_weight=10.0,
        center_patches=False
    )
    
    result_no_center = aligner_no_center.align_patches(
        patches=[copy.deepcopy(p) for p in transformed_patches],
        num_epochs=200,
        learning_rate=0.01
    )
    
    # Test WITH centering
    print("\n--- Testing WITH patch centering ---")
    aligner_with_center = GeoAlignmentProblem2(
        verbose=False, 
        use_orthogonal_reg=True, 
        orthogonal_reg_weight=10.0,
        center_patches=True
    )
    
    result_with_center = aligner_with_center.align_patches(
        patches=[copy.deepcopy(p) for p in transformed_patches],
        num_epochs=200,
        learning_rate=0.01
    )
    
    # Compare results
    from scipy.spatial import procrustes
    
    # Get final embeddings
    embedding_no_center = result_no_center.get_aligned_embedding()
    embedding_with_center = result_with_center.get_aligned_embedding()
    
    # Compute reconstruction quality
    try:
        _, _, procrustes_no_center = procrustes(points, embedding_no_center)
    except:
        procrustes_no_center = float('inf')
        
    try:
        _, _, procrustes_with_center = procrustes(points, embedding_with_center)
    except:
        procrustes_with_center = float('inf')
    
    # Analyze convergence
    final_loss_no_center = result_no_center.loss_history[-1]
    final_loss_with_center = result_with_center.loss_history[-1]
    
    # Find convergence epoch (loss < 1.0)
    convergence_no_center = None
    for i, loss in enumerate(result_no_center.loss_history):
        if loss < 1.0:
            convergence_no_center = i
            break
    
    convergence_with_center = None
    for i, loss in enumerate(result_with_center.loss_history):
        if loss < 1.0:
            convergence_with_center = i
            break
    
    print(f"\n=== COMPARISON RESULTS ===")
    print(f"{'Metric':<25} | {'No Centering':<15} | {'With Centering':<15} | {'Improvement':<15}")
    print(f"{'-'*25} | {'-'*15} | {'-'*15} | {'-'*15}")
    print(f"{'Final Loss':<25} | {final_loss_no_center:<15.6f} | {final_loss_with_center:<15.6f} | {final_loss_no_center/max(final_loss_with_center, 1e-10):<15.1f}x")
    print(f"{'Procrustes Error':<25} | {procrustes_no_center:<15.6f} | {procrustes_with_center:<15.6f} | {procrustes_no_center/max(procrustes_with_center, 1e-10):<15.1f}x")
    
    if convergence_no_center and convergence_with_center:
        print(f"{'Convergence Epoch':<25} | {convergence_no_center:<15} | {convergence_with_center:<15} | {convergence_no_center/convergence_with_center:<15.1f}x")
    elif convergence_with_center:
        print(f"{'Convergence Epoch':<25} | {'No convergence':<15} | {convergence_with_center:<15} | {'N/A':<15}")
    
    # Check orthogonality of learned matrices
    def check_orthogonality(rotations):
        errors = []
        for i, W in enumerate(rotations):
            if i == 0:  # Skip fixed identity
                continue
            WWT = W @ W.T
            I = np.eye(W.shape[0])
            error = np.sum((WWT - I) ** 2)
            errors.append(error)
        return np.mean(errors) if errors else 0
    
    orth_error_no_center = check_orthogonality(result_no_center.rotations)
    orth_error_with_center = check_orthogonality(result_with_center.rotations)
    
    print(f"{'Orthogonality Error':<25} | {orth_error_no_center:<15.6f} | {orth_error_with_center:<15.6f} | {orth_error_no_center/max(orth_error_with_center, 1e-10):<15.1f}x")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    if procrustes_with_center < procrustes_no_center:
        improvement = (procrustes_no_center - procrustes_with_center) / procrustes_no_center * 100
        print(f"✅ Patch centering improves reconstruction by {improvement:.1f}%")
    else:
        print(f"⚠️  Patch centering shows mixed results")
    
    if convergence_with_center and (not convergence_no_center or convergence_with_center < convergence_no_center):
        print(f"✅ Patch centering accelerates convergence")
    
    if orth_error_with_center < orth_error_no_center:
        print(f"✅ Patch centering improves orthogonality")


if __name__ == "__main__":
    test_centering_effect()