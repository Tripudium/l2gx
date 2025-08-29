#!/usr/bin/env python3
"""
Test script for Enhanced MAG240M dataset with lazy loading.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from l2gx.datasets import get_dataset

def test_enhanced_mag240m():
    """Test the enhanced MAG240M dataset."""
    print("Testing Enhanced MAG240M Dataset")
    print("=" * 40)
    
    try:
        # Test recent papers strategy (small subset for testing)
        print("1. Testing recent_papers strategy...")
        dataset = get_dataset(
            "MAG240M-Enhanced", 
            subset_strategy="recent_papers",
            max_papers=1000,
            min_year=2018,
            cache_subsets=True
        )
        
        print(f"Dataset: {dataset}")
        
        # Get statistics
        stats = dataset.get_subset_statistics()
        print(f"Statistics: {stats}")
        
        # Test PyTorch Geometric conversion
        print("\n2. Testing PyTorch Geometric conversion...")
        data = dataset[0]
        print(f"Data object: {data}")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.edge_index.shape[1]}")
        print(f"Feature shape: {data.x.shape}")
        print(f"Label shape: {data.y.shape}")
        
        # Test random papers strategy
        print("\n3. Testing random_papers strategy...")
        dataset2 = get_dataset(
            "MAG240M-Enhanced",
            subset_strategy="random_papers", 
            max_papers=500,
            cache_subsets=True
        )
        
        stats2 = dataset2.get_subset_statistics()
        print(f"Random strategy statistics: {stats2}")
        
        print("\n✅ All tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Note: This test requires the OGB library (pip install ogb)")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_mag240m()
    sys.exit(0 if success else 1)