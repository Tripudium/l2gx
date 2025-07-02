"""
Test the redesigned geo2 aligner.
"""

import copy
import numpy as np
import torch
from scipy.spatial import procrustes
from l2gx.align.geo2 import GeoAlignmentProblem2
from l2gx.patch import Patch
from examples.example import generate_points, voronoi_patches, transform_patches

def test_geo2_basic_functionality():
    """Test basic functionality of the geo2 aligner."""
    print("=== TESTING GEO2 BASIC FUNCTIONALITY ===")
    
    # Create simple test case
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=float)
    
    patches = [
        Patch([0, 1, 4], points[[0, 1, 4]].copy()),      
        Patch([1, 2, 3, 4], points[[1, 2, 3, 4]].copy()) 
    ]
    
    print(f"Created {len(patches)} patches")
    for i, patch in enumerate(patches):
        print(f"  Patch {i}: nodes {patch.nodes}, shape {patch.coordinates.shape}")
    
    # Test patch graph creation
    aligner = GeoAlignmentProblem2(verbose=True, min_overlap=1)
    
    try:
        # This should work with min_overlap=1
        result = aligner.align_patches(
            patches=patches,
            min_overlap=1,
            num_epochs=100,
            learning_rate=0.01,
            device="cpu"
        )
        
        print("✅ Basic alignment completed successfully")
        print(f"Final loss: {result.loss_history[-1]:.6f}")
        
        # Check if transformations were extracted
        print(f"Extracted {len(result.rotations)} rotation matrices")
        print(f"Extracted {len(result.shifts)} translation vectors")
        
        # Check model structure
        print(f"Model has {len(result.model.transformations)} transformation layers")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def test_geo2_patch_graph_connectivity():
    """Test patch graph connectivity checking."""
    print("\n=== TESTING PATCH GRAPH CONNECTIVITY ===")
    
    # Create disconnected patches
    points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    
    # Two patches with no overlap
    patches = [
        Patch([0, 1], points[[0, 1]].copy()),
        Patch([2, 3], points[[2, 3]].copy())
    ]
    
    aligner = GeoAlignmentProblem2(verbose=True)
    
    try:
        aligner.align_patches(patches=patches, min_overlap=1, num_epochs=10)
        print("❌ Should have failed due to disconnected patch graph")
        return False
    except RuntimeError as e:
        if "not connected" in str(e):
            print("✅ Correctly detected disconnected patch graph")
            return True
        else:
            print(f"❌ Wrong error: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_geo2_with_transformed_patches():
    """Test geo2 aligner on transformed Voronoi patches."""
    print("\n=== TESTING GEO2 WITH TRANSFORMED PATCHES ===")
    
    # Generate test data
    np.random.seed(42)
    torch.manual_seed(42)
    
    points = generate_points(n_clusters=3, scale=1.0, std=0.0, max_size=300, min_size=100, dim=2)
    patches, _ = voronoi_patches(points, sample_size=4, min_degree=4, min_overlap=20, min_patch_size=50, eps=1, kmeans=False)
    
    print(f"Generated {len(patches)} patches from {points.shape[0]} points")
    
    # Store original for comparison
    original_patches = [copy.deepcopy(p) for p in patches]
    
    # Apply transformations
    np.random.seed(42)
    transformed_patches = transform_patches([copy.deepcopy(p) for p in patches], shift_scale=2.0, scale_range=None)
    
    print("Applied random transformations")
    
    # Test alignment
    aligner = GeoAlignmentProblem2(verbose=True, min_overlap=20)
    
    try:
        result = aligner.align_patches(
            patches=transformed_patches,
            num_epochs=500,
            learning_rate=0.01,
            device="cpu"
        )
        
        print(f"Alignment completed. Final loss: {result.loss_history[-1]:.6f}")
        
        # Get aligned embedding
        aligned_embedding = result.get_aligned_embedding()
        
        # Compute reconstruction quality
        reconstruction_error = np.linalg.norm(aligned_embedding - points)
        print(f"L2 reconstruction error: {reconstruction_error:.6f}")
        
        try:
            _, _, procrustes_error = procrustes(points, aligned_embedding)
            print(f"Procrustes error: {procrustes_error:.6f}")
            
            if procrustes_error < 0.01:
                print("✅ Excellent reconstruction!")
            elif procrustes_error < 0.1:
                print("✅ Good reconstruction!")
            elif procrustes_error < 1.0:
                print("⚠️  Moderate reconstruction")
            else:
                print("❌ Poor reconstruction")
            
            return procrustes_error
            
        except Exception as e:
            print(f"Procrustes analysis failed: {e}")
            return float('inf')
            
    except Exception as e:
        print(f"❌ Alignment failed: {e}")
        return float('inf')

def test_geo2_vs_geo1_comparison():
    """Compare geo2 aligner with original geo aligner."""
    print("\n=== COMPARING GEO2 VS GEO1 ===")
    
    # Generate test data
    np.random.seed(42)
    torch.manual_seed(42)
    
    points = generate_points(n_clusters=2, scale=1.0, std=0.0, max_size=200, min_size=100, dim=2)
    patches, _ = voronoi_patches(points, sample_size=3, min_degree=4, min_overlap=30, min_patch_size=50, eps=1, kmeans=False)
    
    print(f"Generated {len(patches)} patches from {points.shape[0]} points")
    
    # Apply transformations
    np.random.seed(42)
    transformed_patches = transform_patches([copy.deepcopy(p) for p in patches], shift_scale=1.0, scale_range=None)
    
    # Test geo2
    print("\n--- Testing geo2 ---")
    aligner2 = GeoAlignmentProblem2(verbose=False, min_overlap=30)
    
    try:
        result2 = aligner2.align_patches(
            patches=[copy.deepcopy(p) for p in transformed_patches],
            num_epochs=300,
            learning_rate=0.01
        )
        
        embedding2 = result2.get_aligned_embedding()
        _, _, procrustes2 = procrustes(points, embedding2)
        
        print(f"Geo2: Final loss = {result2.loss_history[-1]:.6f}, Procrustes = {procrustes2:.6f}")
        
    except Exception as e:
        print(f"Geo2 failed: {e}")
        procrustes2 = float('inf')
    
    # Test geo1 (original)
    print("\n--- Testing geo1 ---")
    from l2gx.align.geo import GeoAlignmentProblem
    
    aligner1 = GeoAlignmentProblem(
        num_epochs=300,
        learning_rate=0.01,
        use_orthogonal_reg=True,
        orthogonal_reg_weight=10.0,
        verbose=False,
        min_overlap=30
    )
    
    try:
        aligner1.align_patches([copy.deepcopy(p) for p in transformed_patches], scale=False)
        embedding1 = aligner1.get_aligned_embedding()
        _, _, procrustes1 = procrustes(points, embedding1)
        
        print(f"Geo1: Final loss = {aligner1.loss_hist[-1]:.6f}, Procrustes = {procrustes1:.6f}")
        
    except Exception as e:
        print(f"Geo1 failed: {e}")
        procrustes1 = float('inf')
    
    # Compare results
    print(f"\n--- Comparison ---")
    print(f"Geo1 (original): {procrustes1:.6f}")
    print(f"Geo2 (redesigned): {procrustes2:.6f}")
    
    if procrustes2 < procrustes1:
        improvement = (procrustes1 - procrustes2) / procrustes1 * 100
        print(f"✅ Geo2 is better by {improvement:.1f}%")
    elif procrustes2 < procrustes1 * 1.1:
        print("⚠️  Geo2 performance is comparable")
    else:
        degradation = (procrustes2 - procrustes1) / procrustes1 * 100
        print(f"❌ Geo2 is worse by {degradation:.1f}%")

if __name__ == "__main__":
    # Run tests
    test1_passed = test_geo2_basic_functionality()
    test2_passed = test_geo2_patch_graph_connectivity()
    test3_error = test_geo2_with_transformed_patches()
    test_geo2_vs_geo1_comparison()
    
    print(f"\n=== SUMMARY ===")
    print(f"Basic functionality: {'✅' if test1_passed else '❌'}")
    print(f"Connectivity check: {'✅' if test2_passed else '❌'}")
    print(f"Transformed patches: {'✅' if test3_error < 1.0 else '⚠️' if test3_error < float('inf') else '❌'}")
    
    if test1_passed and test2_passed:
        print("✅ Geo2 aligner basic implementation is working!")
    else:
        print("❌ Geo2 aligner needs fixes")