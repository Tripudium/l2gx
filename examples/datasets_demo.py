"""
L2GX Datasets Demonstration.

This example demonstrates all available datasets in the L2GX framework,
showcasing their capabilities, formats, and usage patterns including:

- Cora: Citation network dataset
- AS-733: Temporal autonomous systems network  
- DGraph: Financial fraud detection graph
- Elliptic: Bitcoin transaction classification
- MAG240M: Large-scale academic citation graph
"""

import numpy as np
import networkx as nx
import time
from typing import Dict, Any
from l2gx.datasets import (
    get_dataset,
    list_available_datasets,
    get_dataset_info,
    DATASET_REGISTRY
)


# ================================
# Dataset Registry Demonstrations
# ================================

def demonstrate_registry_interface():
    """Demonstrate the dataset registry and loading interface."""
    print("🗂️  DATASET REGISTRY INTERFACE")
    print("=" * 60)
    
    # Show available datasets
    datasets = list_available_datasets()
    print(f"📋 Available datasets: {datasets}")
    print()
    
    # Show detailed dataset information
    info = get_dataset_info()
    print("📊 Dataset Information:")
    for name, details in info.items():
        print(f"   {name}:")
        for key, value in details.items():
            print(f"     {key}: {value}")
        print()
    
    # Show registry contents
    print(f"🏷️  Registry contains {len(DATASET_REGISTRY)} dataset classes:")
    for name, cls in DATASET_REGISTRY.items():
        print(f"   {name}: {cls.__name__}")
    print()


# ================================
# Individual Dataset Demonstrations
# ================================

def demonstrate_cora_dataset():
    """Demonstrate the Cora citation network dataset."""
    print("📚 CORA CITATION NETWORK")
    print("=" * 60)
    
    try:
        # Load Cora dataset
        print("🔄 Loading Cora dataset...")
        start_time = time.time()
        cora = get_dataset("Cora")
        load_time = time.time() - start_time
        
        print(f"✅ Loaded in {load_time:.2f}s")
        print(f"📊 Dataset: {cora}")
        print()
        
        # Access different formats
        print("🔀 Format conversions:")
        
        # PyTorch Geometric format (default)
        data = cora[0]
        print(f"   PyG Data: {data}")
        print(f"   Nodes: {data.num_nodes}, Edges: {data.num_edges}")
        print(f"   Features: {data.x.shape}, Labels: {data.y.shape}")
        
        # Raphtory format
        raphtory_graph = cora.to("raphtory")
        print(f"   Raphtory: {raphtory_graph.count_nodes()} nodes, {raphtory_graph.count_edges()} edges")
        
        # Polars format
        edge_df, node_df = cora.to("polars")
        print(f"   Polars: {len(edge_df)} edges, {len(node_df)} nodes")
        print()
        
        # Analyze graph properties
        print("📈 Graph Analysis:")
        G = raphtory_graph.to_networkx()
        
        # Basic properties
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G)
        
        print(f"   Nodes: {num_nodes}")
        print(f"   Edges: {num_edges}")
        print(f"   Density: {density:.4f}")
        
        # Connectivity
        if nx.is_connected(G):
            diameter = nx.diameter(G)
            avg_path = nx.average_shortest_path_length(G)
            print(f"   Diameter: {diameter}")
            print(f"   Avg path length: {avg_path:.2f}")
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            print(f"   Connected components: {nx.number_connected_components(G)}")
            print(f"   Largest CC size: {len(largest_cc)}")
        
        # Degree statistics
        degrees = [d for n, d in G.degree()]
        print(f"   Avg degree: {np.mean(degrees):.2f}")
        print(f"   Max degree: {max(degrees)}")
        print()
        
        return cora
        
    except Exception as e:
        print(f"❌ Failed to load Cora: {e}")
        return None


def demonstrate_as733_dataset():
    """Demonstrate the AS-733 temporal network dataset."""
    print("🌐 AS-733 TEMPORAL NETWORK")
    print("=" * 60)
    
    try:
        # Load AS-733 dataset
        print("🔄 Loading AS-733 dataset...")
        start_time = time.time()
        as733 = get_dataset("as-733")
        load_time = time.time() - start_time
        
        print(f"✅ Loaded in {load_time:.2f}s")
        print(f"📊 Dataset: {as733}")
        print()
        
        # Analyze temporal structure
        print("⏱️  Temporal Analysis:")
        raphtory_graph = as733.to("raphtory")
        
        # Time range
        earliest = raphtory_graph.earliest_date_time
        latest = raphtory_graph.latest_date_time
        print(f"   Time range: {earliest} to {latest}")
        
        # Overall statistics
        total_nodes = raphtory_graph.count_nodes()
        total_edges = raphtory_graph.count_edges()
        total_interactions = raphtory_graph.count_temporal_edges()
        
        print(f"   Total nodes: {total_nodes}")
        print(f"   Unique edges: {total_edges}")
        print(f"   Total interactions: {total_interactions}")
        print()
        
        # Sample time windows
        print("📊 Sample Time Windows:")
        edge_df, node_df = as733.to("polars")
        
        # Get unique timestamps (sample first few)
        timestamps = sorted(edge_df['timestamp'].unique().to_list())[:5]
        for i, ts in enumerate(timestamps):
            window_edges = edge_df.filter(edge_df['timestamp'] == ts)
            print(f"   Window {i+1} ({ts}): {len(window_edges)} edges")
        print()
        
        return as733
        
    except Exception as e:
        print(f"❌ Failed to load AS-733: {e}")
        return None


def demonstrate_dgraph_dataset():
    """Demonstrate the DGraph financial fraud dataset."""
    print("💰 DGRAPH FINANCIAL NETWORK")
    print("=" * 60)
    
    try:
        print("ℹ️  Note: DGraph requires manual download from:")
        print("   https://dgraph.xinye.com/dataset")
        print("   Provide path via source_file parameter")
        print()
        
        # This would work if the file is available
        # dgraph = get_dataset("DGraph", source_file="/path/to/dgraph.zip")
        
        print("📋 DGraph Dataset Features:")
        print("   • Real-world financial transaction graph")
        print("   • Designed for anomaly detection")
        print("   • Large-scale with millions of nodes and edges")
        print("   • Multiple node and edge types")
        print("   • Ground truth labels for evaluation")
        print()
        
        return None
        
    except Exception as e:
        print(f"❌ Failed to load DGraph: {e}")
        return None


def demonstrate_elliptic_dataset():
    """Demonstrate the Elliptic Bitcoin dataset."""
    print("₿  ELLIPTIC BITCOIN TRANSACTIONS")
    print("=" * 60)
    
    try:
        print("ℹ️  Note: Elliptic requires manual download from:")
        print("   https://www.kaggle.com/datasets/ellipticco/elliptic-data-set")
        print("   Provide path via source_file parameter")
        print()
        
        # This would work if the file is available
        # elliptic = get_dataset("Elliptic", source_file="/path/to/elliptic.zip")
        
        print("📋 Elliptic Dataset Features:")
        print("   • Bitcoin transaction classification")
        print("   • 203,769 transactions (nodes)")
        print("   • 234,355 payment flows (edges)")
        print("   • 166 features per transaction")
        print("   • 3 classes: licit (1), illicit (2), unknown (0)")
        print("   • Temporal information available")
        print("   • Used for anti-money laundering research")
        print()
        
        return None
        
    except Exception as e:
        print(f"❌ Failed to load Elliptic: {e}")
        return None


def demonstrate_mag240m_dataset():
    """Demonstrate the MAG240M large-scale dataset."""
    print("🎓 MAG240M ACADEMIC CITATIONS")
    print("=" * 60)
    
    try:
        print("ℹ️  Note: MAG240M requires OGB library:")
        print("   pip install ogb")
        print("   Dataset is very large (~100GB)")
        print()
        
        # This would work if OGB is installed
        # mag240m = get_dataset("MAG240M")
        
        print("📋 MAG240M Dataset Features:")
        print("   • Heterogeneous academic citation graph")
        print("   • 244+ million nodes (papers, authors, institutions, fields)")
        print("   • 1.7+ billion edges (citations, authorship, etc.)")
        print("   • Rich node features and metadata")
        print("   • Publication years from 1800-2020")
        print("   • Multiple node and edge types")
        print("   • Largest public academic graph dataset")
        print()
        
        return None
        
    except Exception as e:
        print(f"❌ Failed to load MAG240M: {e}")
        return None


# ================================
# Format Conversion Demonstrations
# ================================

def demonstrate_format_conversions(dataset):
    """Demonstrate format conversion capabilities."""
    if dataset is None:
        return
    
    print("🔄 FORMAT CONVERSION DEMONSTRATION")
    print("=" * 60)
    
    print(f"📊 Working with: {dataset}")
    print()
    
    # Time different format conversions
    formats = ["polars", "raphtory"]
    
    for fmt in formats:
        print(f"🔀 Converting to {fmt}:")
        start_time = time.time()
        
        try:
            result = dataset.to(fmt)
            conversion_time = time.time() - start_time
            
            if fmt == "polars":
                edge_df, node_df = result
                print(f"   ✅ Success: {len(edge_df)} edges, {len(node_df)} nodes")
            elif fmt == "raphtory":
                print(f"   ✅ Success: {result.count_nodes()} nodes, {result.count_edges()} edges")
            
            print(f"   ⏱️  Conversion time: {conversion_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        print()


# ================================
# Comparison and Analysis
# ================================

def compare_datasets(datasets: Dict[str, Any]):
    """Compare loaded datasets."""
    print("⚔️  DATASET COMPARISON")
    print("=" * 60)
    
    valid_datasets = {name: dataset for name, dataset in datasets.items() if dataset is not None}
    
    if not valid_datasets:
        print("❌ No datasets available for comparison")
        return
    
    print("📊 Loaded Dataset Statistics:")
    for name, dataset in valid_datasets.items():
        print(f"   {name}:")
        
        try:
            # Get Raphtory representation for analysis
            if hasattr(dataset, 'to'):
                graph = dataset.to("raphtory")
                nodes = graph.count_nodes()
                edges = graph.count_edges()
                
                print(f"     Nodes: {nodes:,}")
                print(f"     Edges: {edges:,}")
                
                # Check if temporal
                if hasattr(graph, 'count_temporal_edges'):
                    temporal_edges = graph.count_temporal_edges()
                    if temporal_edges > edges:
                        print(f"     Temporal edges: {temporal_edges:,}")
                        print(f"     Type: Temporal")
                    else:
                        print(f"     Type: Static")
                else:
                    print(f"     Type: Static")
            
        except Exception as e:
            print(f"     Error analyzing: {e}")
        
        print()


# ================================
# Main Demo Runner
# ================================

def run_datasets_demo():
    """Run the complete datasets demonstration."""
    print("🗂️  L2GX COMPREHENSIVE DATASETS DEMONSTRATION")
    print("=" * 80)
    print()
    
    try:
        # 1. Registry interface
        print("PART 1: DATASET REGISTRY")
        print("-" * 40)
        demonstrate_registry_interface()
        
        # 2. Individual dataset demonstrations
        print("PART 2: INDIVIDUAL DATASETS")
        print("-" * 40)
        
        datasets = {}
        
        # Cora (should always work)
        datasets['Cora'] = demonstrate_cora_dataset()
        
        # AS-733 (should work if network is available)
        datasets['AS-733'] = demonstrate_as733_dataset()
        
        # DGraph (requires manual download)
        datasets['DGraph'] = demonstrate_dgraph_dataset()
        
        # Elliptic (requires manual download)
        datasets['Elliptic'] = demonstrate_elliptic_dataset()
        
        # MAG240M (requires OGB library)
        datasets['MAG240M'] = demonstrate_mag240m_dataset()
        
        # 3. Format conversions
        print("PART 3: FORMAT CONVERSIONS")
        print("-" * 40)
        # Use Cora for format conversion demo
        demonstrate_format_conversions(datasets.get('Cora'))
        
        # 4. Dataset comparison
        print("PART 4: DATASET COMPARISON")
        print("-" * 40)
        compare_datasets(datasets)
        
        # Summary
        print("🎊 DEMONSTRATION SUMMARY")
        print("=" * 60)
        print("✅ Successfully demonstrated:")
        print("   • Dataset registry and factory interface")
        print("   • Multiple dataset types: static, temporal, heterogeneous")
        print("   • Format conversion: PyTorch Geometric, Raphtory, Polars")
        print("   • Graph analysis and statistics")
        print("   • Error handling and documentation")
        print()
        
        loaded_count = sum(1 for d in datasets.values() if d is not None)
        print(f"📊 Successfully loaded {loaded_count}/{len(datasets)} datasets")
        print("🚀 The L2GX datasets framework is ready for use!")
        
        return datasets
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_datasets_demo()