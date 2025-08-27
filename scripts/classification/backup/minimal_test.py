#!/usr/bin/env python3
"""
Minimal test to validate the core functionality of each embedding method.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(Path(__file__).parent.parent / "hierarchical"))

from l2gx.datasets import get_dataset
from l2gx.graphs import TGraph
from l2gx.embedding import get_embedding
from binary_hierarchical_embedding import BinaryHierarchicalEmbedding

import warnings
warnings.filterwarnings("ignore")

def test_full_graph_embedding():
    """Test full graph embedding."""
    print("Testing full graph embedding...")
    
    # Load Cora
    dataset = get_dataset("Cora")
    pg_data = dataset.to("torch-geometric")
    graph = TGraph(
        edge_index=pg_data.edge_index,
        x=pg_data.x,
        y=pg_data.y,
        num_nodes=pg_data.num_nodes,
    )
    
    # Create embedder with small parameters
    embedder = get_embedding(
        "vgae",
        embedding_dim=8,
        hidden_dim=16,
        epochs=10,  # Very few epochs for testing
        learning_rate=0.001,
        patience=5,
        verbose=False
    )
    
    start_time = time.time()
    embedding = embedder.fit_transform(graph.to_tg())
    duration = time.time() - start_time
    
    print(f"  Shape: {embedding.shape}, Time: {duration:.2f}s")
    return embedding.shape == (graph.num_nodes, 8)

def test_hierarchical_embedding():
    """Test hierarchical embedding."""
    print("Testing hierarchical embedding...")
    
    # Load Cora
    dataset = get_dataset("Cora")
    pg_data = dataset.to("torch-geometric")
    graph = TGraph(
        edge_index=pg_data.edge_index,
        x=pg_data.x,
        y=pg_data.y,
        num_nodes=pg_data.num_nodes,
    )
    
    # Create hierarchical embedder with minimal parameters
    embedder = BinaryHierarchicalEmbedding(
        max_patch_size=800,
        embedding_dim=8,
        embedding_method="vgae",
        epochs=10,  # Very few epochs for testing
        verbose=False
    )
    
    start_time = time.time()
    
    # Manually set the graph to avoid reloading
    embedder.graph = graph
    embedder.labels = pg_data.y.cpu().numpy()
    
    # Build tree and embed
    embedder.root = embedder.build_hierarchical_tree(graph)
    embedder.embed_leaf_patches()
    embedder.hierarchical_alignment(embedder.root)
    
    embedding = embedder.root.embedding
    duration = time.time() - start_time
    
    print(f"  Shape: {embedding.shape}, Time: {duration:.2f}s")
    return embedding.shape == (graph.num_nodes, 8)

def main():
    """Run minimal tests."""
    print("Running minimal validation tests...")
    print("=" * 50)
    
    # Test 1: Full graph embedding
    try:
        success1 = test_full_graph_embedding()
        print(f"  Full graph embedding: {'✅ PASS' if success1 else '❌ FAIL'}")
    except Exception as e:
        print(f"  Full graph embedding: ❌ FAIL - {e}")
        success1 = False
    
    # Test 2: Hierarchical embedding  
    try:
        success2 = test_hierarchical_embedding()
        print(f"  Hierarchical embedding: {'✅ PASS' if success2 else '❌ FAIL'}")
    except Exception as e:
        print(f"  Hierarchical embedding: ❌ FAIL - {e}")
        success2 = False
    
    print("=" * 50)
    
    if success1 and success2:
        print("✅ All core methods working! Ready for full experiment.")
        return True
    else:
        print("❌ Some methods failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)