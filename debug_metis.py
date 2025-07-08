#!/usr/bin/env python3
"""
Debug script for METIS clustering issue with Cora dataset
"""

import sys
import traceback
sys.path.append('.')

import torch
import numpy as np
from l2gx.datasets import get_dataset
from l2gx.graphs import TGraph
from l2gx.patch import generate_patches
from l2gx.patch.clustering.metis import metis_clustering

def debug_metis_issue():
    """Debug the METIS clustering issue step by step."""
    
    print("="*60)
    print("DEBUGGING METIS CLUSTERING ISSUE")
    print("="*60)
    
    # Step 1: Load Cora dataset
    print("\n1. Loading Cora dataset...")
    try:
        cora = get_dataset("Cora")
        data = cora.to("torch-geometric")
        print(f"   ✅ Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges")
    except Exception as e:
        print(f"   ❌ Failed to load dataset: {e}")
        return
    
    # Step 2: Create TGraph
    print("\n2. Creating TGraph...")
    try:
        tg = TGraph(data.edge_index, edge_attr=data.edge_attr, x=data.x)
        print(f"   ✅ TGraph created: {tg.num_nodes} nodes, {tg.num_edges} edges")
        print(f"   Adjacency index shape: {tg.adj_index.shape}")
        print(f"   Edge index shape: {tg.edge_index.shape}")
        print(f"   Device: {tg.device}")
        print(f"   Edge index dtype: {tg.edge_index.dtype}")
        print(f"   Adj index dtype: {tg.adj_index.dtype}")
    except Exception as e:
        print(f"   ❌ Failed to create TGraph: {e}")
        traceback.print_exc()
        return
    
    # Step 3: Check TGraph properties for METIS compatibility
    print("\n3. Checking TGraph properties...")
    try:
        print(f"   Adjacency index range: [{tg.adj_index.min()}, {tg.adj_index.max()}]")
        print(f"   Edge index[1] range: [{tg.edge_index[1].min()}, {tg.edge_index[1].max()}]")
        print(f"   Number of edges in adjacency list: {len(tg.edge_index[1])}")
        print(f"   Expected edges from adj_index: {tg.adj_index[-1]}")
        
        # Check if adjacency structure is valid
        if tg.adj_index[-1] != tg.num_edges:
            print(f"   ⚠️  Warning: adj_index[-1] ({tg.adj_index[-1]}) != num_edges ({tg.num_edges})")
        else:
            print(f"   ✅ Adjacency structure looks correct")
            
        # Check for any negative indices
        if (tg.edge_index < 0).any():
            print(f"   ❌ Found negative indices in edge_index")
        else:
            print(f"   ✅ No negative indices in edge_index")
            
    except Exception as e:
        print(f"   ❌ Error checking properties: {e}")
        traceback.print_exc()
    
    # Step 4: Test METIS clustering directly
    print("\n4. Testing METIS clustering directly...")
    try:
        clusters = metis_clustering(tg, num_clusters=10)
        print(f"   ✅ METIS clustering successful!")
        print(f"   Cluster tensor shape: {clusters.shape}")
        print(f"   Unique clusters: {len(torch.unique(clusters))}")
        print(f"   Cluster range: [{clusters.min()}, {clusters.max()}]")
    except Exception as e:
        print(f"   ❌ METIS clustering failed: {e}")
        traceback.print_exc()
        
        # Try to diagnose the specific issue
        print("\n   Diagnosing METIS issue...")
        try:
            import pymetis
            print(f"   PyMETIS available: {pymetis}")
            
            # Check data types for PyMETIS compatibility
            print(f"   adj_index type: {type(tg.adj_index)} {tg.adj_index.dtype}")
            print(f"   edge_index[1] type: {type(tg.edge_index[1])} {tg.edge_index[1].dtype}")
            
            # Convert to CPU and numpy for compatibility
            adj_index_cpu = tg.adj_index.cpu().numpy().astype(np.int32)
            edge_list_cpu = tg.edge_index[1].cpu().numpy().astype(np.int32)
            
            print(f"   Converted adj_index: {adj_index_cpu.dtype} shape {adj_index_cpu.shape}")
            print(f"   Converted edge_list: {edge_list_cpu.dtype} shape {edge_list_cpu.shape}")
            
            # Try PyMETIS directly
            print("   Trying PyMETIS directly...")
            edge_cuts, memberships = pymetis.part_graph(
                10,  # num_clusters
                adjncy=edge_list_cpu,
                xadj=adj_index_cpu,
                eweights=None
            )
            print(f"   ✅ PyMETIS direct call successful!")
            print(f"   Edge cuts: {edge_cuts}")
            print(f"   Memberships length: {len(memberships)}")
            
        except Exception as inner_e:
            print(f"   ❌ PyMETIS direct call failed: {inner_e}")
            traceback.print_exc()
    
    # Step 5: Test generate_patches with METIS
    print("\n5. Testing generate_patches with METIS...")
    try:
        patches, patch_graph = generate_patches(tg, num_patches=10, clustering_method='metis')
        print(f"   ✅ generate_patches successful!")
        print(f"   Generated {len(patches)} patches")
        patch_sizes = [len(p.nodes) for p in patches]
        print(f"   Patch sizes: min={min(patch_sizes)}, max={max(patch_sizes)}, avg={np.mean(patch_sizes):.1f}")
    except Exception as e:
        print(f"   ❌ generate_patches failed: {e}")
        traceback.print_exc()
    
    # Step 6: Try alternative clustering methods
    print("\n6. Testing alternative clustering methods...")
    for method in ['louvain', 'fennel']:
        try:
            print(f"   Testing {method}...")
            patches, patch_graph = generate_patches(tg, num_patches=10, clustering_method=method)
            print(f"   ✅ {method} successful! Generated {len(patches)} patches")
        except Exception as e:
            print(f"   ❌ {method} failed: {e}")
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)

if __name__ == "__main__":
    debug_metis_issue()