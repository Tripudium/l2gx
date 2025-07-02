"""
Example demonstrating the fix for orthogonal regularization with translated patches.

This shows how the new centering and scale-preserving features solve the
"squashing" problem when patches have both rotations and translations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from l2gx.align import GeoAlignmentProblem
from l2gx.patch import Patch

def create_test_patches_with_translation():
    """Create test patches with both rotations AND translations."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Original coordinates - create a recognizable shape
    n_points = 30
    t = np.linspace(0, 2*np.pi, n_points)
    coords = np.column_stack([2*np.cos(t), np.sin(t)])  # Ellipse shape
    
    patches = []
    
    # Patch 0: Identity (reference)
    patches.append(Patch(range(n_points), coords.copy()))
    
    # Patch 1: 30-degree rotation + large translation
    theta = np.pi / 6
    R1 = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
    coords1 = coords @ R1.T + np.array([5.0, 3.0])  # Large translation
    patches.append(Patch(range(n_points), coords1))
    
    # Patch 2: 90-degree rotation + different translation
    theta = np.pi / 2
    R2 = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
    coords2 = coords @ R2.T + np.array([-3.0, 4.0])  # Different translation
    patches.append(Patch(range(n_points), coords2))
    
    # Patch 3: -45-degree rotation + another translation
    theta = -np.pi / 4
    R3 = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
    coords3 = coords @ R3.T + np.array([2.0, -2.0])
    patches.append(Patch(range(n_points), coords3))
    
    return patches

def plot_patches_comparison(original_patches, aligned_patches_old, aligned_patches_new):
    """Plot comparison of original, old method, and new method."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['red', 'blue', 'green', 'orange']
    
    # Original patches
    axes[0].set_title('Original Patches\n(with translations)')
    for i, patch in enumerate(original_patches):
        axes[0].scatter(patch.coordinates[:, 0], patch.coordinates[:, 1], 
                       c=colors[i], alpha=0.7, label=f'Patch {i}')
    axes[0].set_aspect('equal')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Old method (without centering)
    axes[1].set_title('Aligned: No Centering\n(may show squashing)')
    for i, patch in enumerate(aligned_patches_old):
        axes[1].scatter(patch.coordinates[:, 0], patch.coordinates[:, 1], 
                       c=colors[i], alpha=0.7, label=f'Patch {i}')
    axes[1].set_aspect('equal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # New method (with centering)
    axes[2].set_title('Aligned: With Centering\n(should preserve shape)')
    for i, patch in enumerate(aligned_patches_new):
        axes[2].scatter(patch.coordinates[:, 0], patch.coordinates[:, 1], 
                       c=colors[i], alpha=0.7, label=f'Patch {i}')
    axes[2].set_aspect('equal')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def test_translation_fix():
    """Test the fix for the translation issue."""
    patches = create_test_patches_with_translation()
    
    print("=== Testing OLD method (no centering, strict orthogonality) ===")
    aligner_old = GeoAlignmentProblem(
        num_epochs=300,
        learning_rate=0.01,
        model_type="affine",
        use_orthogonal_reg=True,
        orthogonal_reg_weight=0.1,
        center_patches=False,  # OLD: No centering
        preserve_scale=False,   # OLD: Strict orthogonality
        verbose=True
    )
    
    # Make copies for fair comparison
    patches_old = [Patch(p.nodes.copy(), p.coordinates.copy()) for p in patches]
    aligner_old.align_patches(patches_old)
    
    print("\n" + "="*70)
    print("=== Testing NEW method (with centering, scale-preserving) ===")
    aligner_new = GeoAlignmentProblem(
        num_epochs=300,
        learning_rate=0.01,
        model_type="affine",
        use_orthogonal_reg=True,
        orthogonal_reg_weight=10.0, # Higher regularization weight  
        center_patches=True,        # NEW: Center patches first
        preserve_scale=False,       # Strict orthogonality for better results
        verbose=True
    )
    
    # Make copies for fair comparison  
    patches_new = [Patch(p.nodes.copy(), p.coordinates.copy()) for p in patches]
    aligner_new.align_patches(patches_new)
    
    print("\n=== Orthogonality Analysis ===")
    print("Old method (no centering):")
    metrics_old = aligner_old.check_orthogonality(aligner_old.model)
    for patch_name, metrics in metrics_old.items():
        if patch_name != 'patch_0':  # Skip fixed patch
            print(f"  {patch_name}: Frobenius error = {metrics['frobenius_error']:.4f}, "
                  f"Det = {metrics['determinant']:.4f}")
    
    print("\nNew method (with centering):")
    metrics_new = aligner_new.check_orthogonality(aligner_new.model)
    for patch_name, metrics in metrics_new.items():
        if patch_name != 'patch_0':  # Skip fixed patch
            print(f"  {patch_name}: Frobenius error = {metrics['frobenius_error']:.4f}, "
                  f"Det = {metrics['determinant']:.4f}")
    
    # Visual comparison
    print("\n=== Visual Comparison ===")
    try:
        plot_patches_comparison(patches, patches_old, patches_new)
        print("Plots displayed successfully!")
    except Exception as e:
        print(f"Plotting failed (this is normal in headless environments): {e}")
    
    print("\n=== Summary ===")
    print("The NEW method should show:")
    print("1. Better preserved ellipse shapes (less squashing)")
    print("2. Good orthogonality metrics")
    print("3. Successful alignment despite large translations")

if __name__ == "__main__":
    test_translation_fix()