#!/usr/bin/env python3
"""
Example usage of Enhanced MAG240M dataset with different sampling strategies.

This example demonstrates how to:
1. Load subsets using different strategies
2. Extract paper features and citations
3. Create embeddings on manageable subsets
4. Analyze the results

Note: Requires OGB library (pip install ogb)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from l2gx.datasets import get_dataset


def demonstrate_strategies():
    """Demonstrate different subset extraction strategies."""
    
    print("MAG240M Enhanced Dataset Examples")
    print("=" * 50)
    
    strategies = [
        {
            "name": "Recent Papers (CS focused)",
            "params": {
                "subset_strategy": "recent_papers",
                "max_papers": 10000,
                "min_year": 2018,
                "max_year": 2023
            }
        },
        {
            "name": "Random Sample",
            "params": {
                "subset_strategy": "random_papers", 
                "max_papers": 5000
            }
        },
        {
            "name": "Temporal Window",
            "params": {
                "subset_strategy": "temporal_window",
                "max_papers": 8000,
                "min_year": 2015,
                "max_year": 2017
            }
        },
        {
            "name": "Highly Cited Papers",
            "params": {
                "subset_strategy": "citation_subgraph",
                "max_papers": 3000
            }
        }
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{i}. {strategy['name']}")
        print("-" * 30)
        
        try:
            # Create dataset with strategy
            dataset = get_dataset("MAG240M-Enhanced", **strategy["params"])
            
            # Get statistics
            stats = dataset.get_subset_statistics()
            print(f"Subset size: {stats.get('num_nodes', 0):,} nodes, {stats.get('num_edges', 0):,} edges")
            
            if 'year_range' in stats:
                year_range = stats['year_range']
                print(f"Year range: {year_range['min']} - {year_range['max']} ({year_range['count']} papers)")
            
            if 'node_types' in stats:
                print(f"Node types: {stats['node_types']}")
            
            # Convert to PyTorch Geometric
            data = dataset[0]
            print(f"PyG Data: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
            print(f"Features: {data.x.shape}, Labels: {data.y.shape}")
            
        except Exception as e:
            print(f"⚠️  Error with {strategy['name']}: {e}")


def embedding_pipeline_example():
    """Example of using MAG240M subset in an embedding pipeline."""
    
    print("\n" + "=" * 50)
    print("Embedding Pipeline Example")
    print("=" * 50)
    
    try:
        # Load a manageable subset
        dataset = get_dataset(
            "MAG240M-Enhanced",
            subset_strategy="recent_papers",
            max_papers=20000,
            min_year=2016,
            cache_subsets=True
        )
        
        print(f"Loaded dataset: {dataset}")
        
        # Get the graph data
        data = dataset[0]
        print(f"Graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
        
        # Here you would typically:
        # 1. Convert to TGraph for L2GX compatibility
        # from l2gx.graphs.tgraph import TGraph
        # tgraph = TGraph.from_pyg(data)
        
        # 2. Apply patching strategy
        # from l2gx.patch.generate import generate_patches
        # patches = generate_patches(tgraph, method="fennel", num_patches=10)
        
        # 3. Train local embeddings on patches
        # from l2gx.embedding import get_embedding_method
        # embedder = get_embedding_method("VGAE")
        # embeddings = [embedder.embed(patch.subgraph) for patch in patches]
        
        # 4. Align local embeddings to global
        # from l2gx.align import get_alignment_method  
        # aligner = get_alignment_method("L2G")
        # global_embedding = aligner.align(embeddings, patches)
        
        print("✅ Pipeline setup complete (embedding code commented out)")
        
    except Exception as e:
        print(f"❌ Pipeline example failed: {e}")


def main():
    """Run all examples."""
    
    try:
        # Check if OGB is available
        from ogb.lsc import MAG240MDataset
        print("OGB library found - running full examples")
        
        demonstrate_strategies()
        embedding_pipeline_example()
        
    except ImportError:
        print("⚠️  OGB library not found")
        print("Install with: pip install ogb")
        print("Skipping MAG240M examples")
        
        # Show conceptual example instead
        print("\nConceptual Usage:")
        print("```python")
        print("from l2gx.datasets import get_dataset")
        print("")
        print("# Load recent CS papers")
        print('dataset = get_dataset("MAG240M-Enhanced",')
        print('                     subset_strategy="recent_papers",')
        print('                     max_papers=50000,')
        print('                     min_year=2015)')
        print("")
        print("# Get PyTorch Geometric data")
        print("data = dataset[0]")
        print("print(f'Loaded {data.num_nodes} nodes, {data.edge_index.shape[1]} edges')")
        print("```")


if __name__ == "__main__":
    main()