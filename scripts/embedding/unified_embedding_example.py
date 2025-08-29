#!/usr/bin/env python3
"""
Unified Embedding Framework Examples

Demonstrates how to use the new unified embedding framework with:
- get_embedding("patched", ...)
- get_embedding("hierarchical", ...)
- Configuration-driven experiments
"""

import warnings
from pathlib import Path

import numpy as np

from l2gx.align import get_aligner
from l2gx.datasets import get_dataset
from l2gx.embedding import get_embedding

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def example_1_patched_l2g():
    """Example 1: Patched embedding with L2G alignment."""
    print("EXAMPLE 1: Patched Embedding with L2G Alignment")
    print("=" * 60)
    
    # Load dataset
    dataset = get_dataset("Cora")
    data = dataset.to("torch-geometric")
    
    print(f"Dataset: {data.num_nodes} nodes, {data.num_edges} edges")
    
    # Create L2G aligner
    l2g_aligner = get_aligner("l2g")
    l2g_aligner.randomized_method = "randomized"
    l2g_aligner.sketch_method = "rademacher"
    
    # Create patched embedder using unified framework
    embedder = get_embedding(
        "patched",
        embedding_dim=64,
        aligner=l2g_aligner,
        num_patches=8,
        base_method="vgae",
        clustering_method="metis",
        min_overlap=10,
        target_overlap=20,
        epochs=100,
        verbose=True
    )
    
    # Compute embedding
    embedding = embedder.fit_transform(data)
    
    print(f"Generated embedding shape: {embedding.shape}")
    print(f"Patch info: {embedder.get_patch_info()}")
    print()
    
    return embedding


def example_2_hierarchical():
    """Example 2: Hierarchical embedding with smart alignment."""
    print("EXAMPLE 2: Hierarchical Embedding with Smart Alignment")
    print("=" * 60)
    
    # Load dataset
    dataset = get_dataset("Cora")
    data = dataset.to("torch-geometric")
    
    print(f"Dataset: {data.num_nodes} nodes, {data.num_edges} edges")
    
    # Create aligner for multi-way trees
    l2g_aligner = get_aligner("l2g")
    l2g_aligner.randomized_method = "randomized" 
    l2g_aligner.sketch_method = "gaussian"
    
    # Create hierarchical embedder
    embedder = get_embedding(
        "hierarchical",
        embedding_dim=64,
        aligner=l2g_aligner,
        max_patch_size=600,
        base_method="vgae",
        min_overlap=32,
        target_overlap=64,
        epochs=100,
        verbose=True
    )
    
    # Compute embedding
    embedding = embedder.fit_transform(data)
    
    print(f"Generated embedding shape: {embedding.shape}")
    print(f"Patch info: {embedder.get_patch_info()}")
    print()
    
    return embedding


def example_3_patched_geo():
    """Example 3: Patched embedding with Geo alignment."""
    print("EXAMPLE 3: Patched Embedding with Geo Alignment")
    print("=" * 60)
    
    # Load dataset
    dataset = get_dataset("Cora")
    data = dataset.to("torch-geometric")
    
    print(f"Dataset: {data.num_nodes} nodes, {data.num_edges} edges")
    
    # Create Geo aligner
    geo_aligner = get_aligner(
        "geo",
        method="orthogonal",
        use_scale=True,
        use_randomized_init=True,
        randomized_method="randomized",
        verbose=False
    )
    
    # Create patched embedder using unified framework
    embedder = get_embedding(
        "patched",
        embedding_dim=64,
        aligner=geo_aligner,
        num_patches=6,
        base_method="svd",  # Use SVD for faster example
        clustering_method="metis",
        min_overlap=10,
        target_overlap=20,
        verbose=True
    )
    
    # Compute embedding
    embedding = embedder.fit_transform(data)
    
    print(f"Generated embedding shape: {embedding.shape}")
    print(f"Patch info: {embedder.get_patch_info()}")
    print()
    
    return embedding


def example_4_config_driven():
    """Example 4: Configuration-driven experiment."""
    print("EXAMPLE 4: Configuration-Driven Experiment")
    print("=" * 60)
    
    # Import experiment class
    from run_embedding_config import ConfigurableEmbeddingExperiment
    
    # Run experiment with unified config
    config_path = "config/unified_patched_l2g_config.yaml"
    
    if Path(config_path).exists():
        experiment = ConfigurableEmbeddingExperiment(config_path)
        embedding, patches, data = experiment.run_experiment()
        
        print(f"Config-driven embedding shape: {embedding.shape}")
        
        return embedding
    else:
        print(f"Configuration file not found: {config_path}")
        print("You can create one using the provided templates")
        return None


def main():
    """Run all examples."""
    print("UNIFIED EMBEDDING FRAMEWORK EXAMPLES")
    print("=" * 80)
    print()
    
    try:
        # Run examples
        emb1 = example_1_patched_l2g()
        emb2 = example_2_hierarchical() 
        emb3 = example_3_patched_geo()
        emb4 = example_4_config_driven()
        
        print("=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("✓ Patched L2G embedding")
        print("✓ Hierarchical embedding") 
        print("✓ Patched Geo embedding")
        if emb4 is not None:
            print("✓ Configuration-driven experiment")
        
        print()
        print("Key benefits of unified framework:")
        print("- Consistent API across all embedding methods")
        print("- Automatic aligner configuration and parameter extraction")
        print("- Smart alignment selection for hierarchical embeddings")
        print("- Configuration-driven experiments for reproducibility")
        
    except Exception as e:
        print(f"Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()