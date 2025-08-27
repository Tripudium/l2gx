#!/usr/bin/env python3
"""
Troubleshooting script for ARPACK errors in L2GX.

This script demonstrates several strategies to avoid or handle ARPACK errors
when using L2G alignment with patched embeddings.
"""

import sys
from pathlib import Path

# Add parent directory to path for L2GX imports if running directly
if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))

import numpy as np
import warnings
from l2gx.datasets import get_dataset
from l2gx.embedding import get_embedding
from l2gx.align import get_aligner
from l2gx.graphs import TGraph

def strategy_1_use_geo_alignment():
    """Strategy 1: Use Geo alignment instead of L2G (more stable)."""
    print("\n" + "="*60)
    print("STRATEGY 1: Use Geo alignment (more numerically stable)")
    print("="*60)
    
    dataset = get_dataset("Cora")
    data = TGraph.from_tg(dataset.to("torch-geometric"))
    
    # Geo alignment is generally more stable than L2G
    geo_aligner = get_aligner("geo", method="orthogonal", use_scale=True)
    
    embedder = get_embedding(
        "patched",
        embedding_dim=64,
        aligner=geo_aligner,
        num_patches=8,
        base_method="vgae",
        epochs=50,
        verbose=False
    )
    
    embedding = embedder.fit_transform(data.to_tg())
    print(f"✅ Geo alignment completed successfully!")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Mean norm: {np.mean(np.linalg.norm(embedding, axis=1)):.3f}")

def strategy_2_reduce_patches():
    """Strategy 2: Use fewer patches to reduce complexity."""
    print("\n" + "="*60) 
    print("STRATEGY 2: Reduce number of patches")
    print("="*60)
    
    dataset = get_dataset("Cora")
    data = TGraph.from_tg(dataset.to("torch-geometric"))
    
    l2g_aligner = get_aligner("l2g")
    l2g_aligner.randomized_method = "standard"  # Less randomization
    
    embedder = get_embedding(
        "patched",
        embedding_dim=64,
        aligner=l2g_aligner, 
        num_patches=4,        # Fewer patches = more stable
        base_method="svd",    # SVD is deterministic
        verbose=False
    )
    
    embedding = embedder.fit_transform(data.to_tg())
    print(f"✅ L2G with fewer patches completed!")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Mean norm: {np.mean(np.linalg.norm(embedding, axis=1)):.3f}")

def strategy_3_increase_overlap():
    """Strategy 3: Increase patch overlap for better numerical conditioning."""
    print("\n" + "="*60)
    print("STRATEGY 3: Increase patch overlap")
    print("="*60)
    
    dataset = get_dataset("Cora") 
    data = TGraph.from_tg(dataset.to("torch-geometric"))
    
    l2g_aligner = get_aligner("l2g")
    l2g_aligner.randomized_method = "standard"
    
    embedder = get_embedding(
        "patched",
        embedding_dim=64,
        aligner=l2g_aligner,
        num_patches=6,
        base_method="svd",
        min_overlap=30,       # Larger overlap
        target_overlap=60,    # Much larger overlap
        verbose=False
    )
    
    embedding = embedder.fit_transform(data.to_tg())
    print(f"✅ L2G with increased overlap completed!")
    print(f"   Embedding shape: {embedding.shape}") 
    print(f"   Mean norm: {np.mean(np.linalg.norm(embedding, axis=1)):.3f}")

def strategy_4_use_deterministic():
    """Strategy 4: Use deterministic methods to avoid randomization issues."""
    print("\n" + "="*60)
    print("STRATEGY 4: Use deterministic methods")
    print("="*60)
    
    dataset = get_dataset("Cora")
    data = TGraph.from_tg(dataset.to("torch-geometric"))
    
    l2g_aligner = get_aligner("l2g")
    l2g_aligner.randomized_method = "standard"  # No randomization
    l2g_aligner.sketch_method = "gaussian"       # Standard sketching
    
    embedder = get_embedding(
        "patched",
        embedding_dim=32,     # Lower dimension = less complexity
        aligner=l2g_aligner,
        num_patches=6,
        base_method="svd",    # Deterministic method
        verbose=False
    )
    
    embedding = embedder.fit_transform(data.to_tg())
    print(f"✅ Deterministic L2G completed!")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Mean norm: {np.mean(np.linalg.norm(embedding, axis=1)):.3f}")

def strategy_5_test_fallback():
    """Strategy 5: Test the automatic fallback mechanism."""
    print("\n" + "="*60)
    print("STRATEGY 5: Test automatic fallback (if ARPACK fails)")
    print("="*60)
    
    dataset = get_dataset("Cora")
    data = TGraph.from_tg(dataset.to("torch-geometric"))
    
    # Create conditions that might trigger ARPACK errors
    l2g_aligner = get_aligner("l2g") 
    l2g_aligner.randomized_method = "randomized"
    l2g_aligner.sketch_method = "rademacher"
    
    embedder = get_embedding(
        "patched",
        embedding_dim=128,    # High dimension
        aligner=l2g_aligner,
        num_patches=10,       # Many patches
        base_method="vgae",   # Can create problematic matrices
        min_overlap=10,       # Small overlap
        target_overlap=20,
        epochs=50,
        verbose=True          # Show fallback messages
    )
    
    embedding = embedder.fit_transform(data.to_tg())
    print(f"✅ L2G with fallback handling completed!")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Mean norm: {np.mean(np.linalg.norm(embedding, axis=1)):.3f}")

def main():
    """Run all troubleshooting strategies."""
    print("L2GX ARPACK Error Troubleshooting Guide")
    print("=" * 60)
    print("This script demonstrates strategies to handle ARPACK errors in L2G alignment.")
    print("If you encounter ARPACK errors, try these approaches in order:")
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    try:
        strategy_1_use_geo_alignment()
        strategy_2_reduce_patches()
        strategy_3_increase_overlap()
        strategy_4_use_deterministic()
        strategy_5_test_fallback()
        
        print("\n" + "="*60)
        print("SUMMARY: All strategies completed successfully!")
        print("="*60)
        print("\nIf you're still experiencing ARPACK errors:")
        print("1. Try Geo alignment (Strategy 1) - usually most reliable")
        print("2. Reduce the number of patches (Strategy 2)")  
        print("3. Use SVD instead of neural methods for base embeddings")
        print("4. Increase patch overlap for better numerical conditioning")
        print("5. The improved L2G aligner now has automatic fallback to Procrustes")
        
    except Exception as e:
        print(f"\n❌ Error in troubleshooting: {e}")
        print("This suggests a deeper issue that needs investigation.")

if __name__ == "__main__":
    main()