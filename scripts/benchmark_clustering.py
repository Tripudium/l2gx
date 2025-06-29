#!/usr/bin/env python3
"""
Clustering Algorithm Benchmarking Script

This script benchmarks different clustering implementations to compare
performance and validate correctness of the Rust implementations.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys

# Add L2G to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from l2gv2.graphs import TGraph
from l2gv2.patch.clustering.fennel import fennel_clustering_safe
from l2gv2.patch.clustering.rust_fennel import (
    fennel_clustering_rust, 
    is_rust_available,
    benchmark_rust_vs_python
)


def generate_test_graph(num_nodes: int, avg_degree: int = 10, seed: int = 42) -> TGraph:
    """Generate a random test graph with specified properties"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate random edges
    num_edges = (num_nodes * avg_degree) // 2
    
    # Create edges with some structure (not completely random)
    edges = []
    
    # Add some structured clusters
    cluster_size = num_nodes // 10
    for cluster_start in range(0, num_nodes, cluster_size):
        cluster_end = min(cluster_start + cluster_size, num_nodes)
        
        # Dense connections within cluster
        for i in range(cluster_start, cluster_end):
            for j in range(i + 1, min(i + avg_degree // 2, cluster_end)):
                if np.random.random() < 0.7:  # 70% chance of edge within cluster
                    edges.append([i, j])
    
    # Add some random inter-cluster edges
    for _ in range(num_edges // 3):
        i = np.random.randint(0, num_nodes)
        j = np.random.randint(0, num_nodes)
        if i != j:
            edges.append([i, j])
    
    # Remove duplicates and convert to tensor
    edges = list(set(tuple(sorted(edge)) for edge in edges))
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    # Make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    return TGraph(edge_index, num_nodes=num_nodes)


def benchmark_single_size(num_nodes: int, num_clusters: int, num_runs: int = 3) -> dict:
    """Benchmark clustering algorithms on a single graph size"""
    print(f"\nBenchmarking {num_nodes} nodes, {num_clusters} clusters...")
    
    # Generate test graph
    graph = generate_test_graph(num_nodes)
    print(f"Generated graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    results = {
        'num_nodes': num_nodes,
        'num_clusters': num_clusters,
        'num_edges': graph.num_edges,
    }
    
    # Python/Numba Fennel (baseline)
    python_times = []
    try:
        edge_index_np = graph.edge_index.cpu().numpy()
        adj_index_np = graph.adj_index.cpu().numpy()
        
        for run in range(num_runs):
            print(f"  Python run {run + 1}/{num_runs}...", end=" ")
            start_time = time.time()
            
            python_clusters = fennel_clustering_safe(
                edge_index_np,
                adj_index_np,
                graph.num_nodes,
                num_clusters,
                verbose=False
            )
            
            end_time = time.time()
            python_times.append(end_time - start_time)
            print(f"{python_times[-1]:.3f}s")
        
        results['python_times'] = python_times
        results['python_mean'] = np.mean(python_times)
        results['python_std'] = np.std(python_times)
        results['python_clusters'] = len(np.unique(python_clusters[python_clusters >= 0]))
        
    except Exception as e:
        print(f"  Python benchmark failed: {e}")
        results['python_error'] = str(e)
    
    # Rust Fennel (single-threaded)
    if is_rust_available():
        rust_times = []
        try:
            for run in range(num_runs):
                print(f"  Rust run {run + 1}/{num_runs}...", end=" ")
                start_time = time.time()
                
                rust_clusters = fennel_clustering_rust(
                    graph,
                    num_clusters,
                    parallel=False,
                    verbose=False
                )
                
                end_time = time.time()
                rust_times.append(end_time - start_time)
                print(f"{rust_times[-1]:.3f}s")
            
            results['rust_times'] = rust_times
            results['rust_mean'] = np.mean(rust_times)
            results['rust_std'] = np.std(rust_times)
            results['rust_clusters'] = len(torch.unique(rust_clusters[rust_clusters >= 0]))
            
            if 'python_mean' in results:
                results['speedup'] = results['python_mean'] / results['rust_mean']
            
        except Exception as e:
            print(f"  Rust benchmark failed: {e}")
            results['rust_error'] = str(e)
        
        # Rust Fennel (parallel)
        if num_nodes >= 1000:  # Only test parallel for larger graphs
            rust_parallel_times = []
            try:
                for run in range(num_runs):
                    print(f"  Rust parallel run {run + 1}/{num_runs}...", end=" ")
                    start_time = time.time()
                    
                    rust_parallel_clusters = fennel_clustering_rust(
                        graph,
                        num_clusters,
                        parallel=True,
                        verbose=False
                    )
                    
                    end_time = time.time()
                    rust_parallel_times.append(end_time - start_time)
                    print(f"{rust_parallel_times[-1]:.3f}s")
                
                results['rust_parallel_times'] = rust_parallel_times
                results['rust_parallel_mean'] = np.mean(rust_parallel_times)
                results['rust_parallel_std'] = np.std(rust_parallel_times)
                results['rust_parallel_clusters'] = len(torch.unique(rust_parallel_clusters[rust_parallel_clusters >= 0]))
                
                if 'python_mean' in results:
                    results['parallel_speedup'] = results['python_mean'] / results['rust_parallel_mean']
                
            except Exception as e:
                print(f"  Rust parallel benchmark failed: {e}")
                results['rust_parallel_error'] = str(e)
    else:
        print("  Rust implementation not available")
        results['rust_error'] = "Rust implementation not available"
    
    return results


def run_scaling_benchmark():
    """Run benchmark across different graph sizes"""
    # Test different graph sizes
    test_configs = [
        (100, 5),      # Small graph
        (500, 10),     # Medium graph
        (1000, 20),    # Large graph
        (2000, 30),    # Very large graph
        (5000, 50),    # Huge graph (if system can handle it)
    ]
    
    all_results = []
    
    print("=" * 60)
    print("SCALING BENCHMARK")
    print("=" * 60)
    
    for num_nodes, num_clusters in test_configs:
        try:
            results = benchmark_single_size(num_nodes, num_clusters, num_runs=3)
            all_results.append(results)
            
            # Print summary
            if 'python_mean' in results and 'rust_mean' in results:
                speedup = results['speedup']
                print(f"  → Speedup: {speedup:.2f}x")
                if 'parallel_speedup' in results:
                    parallel_speedup = results['parallel_speedup']
                    print(f"  → Parallel speedup: {parallel_speedup:.2f}x")
            
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user")
            break
        except Exception as e:
            print(f"Benchmark failed for {num_nodes} nodes: {e}")
            continue
    
    return all_results


def plot_results(results):
    """Create performance comparison plots"""
    if not results:
        print("No results to plot")
        return
    
    # Extract data for plotting
    node_counts = [r['num_nodes'] for r in results]
    python_times = [r.get('python_mean', np.nan) for r in results]
    rust_times = [r.get('rust_mean', np.nan) for r in results]
    rust_parallel_times = [r.get('rust_parallel_mean', np.nan) for r in results]
    speedups = [r.get('speedup', np.nan) for r in results]
    parallel_speedups = [r.get('parallel_speedup', np.nan) for r in results]
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Execution times
    ax1.plot(node_counts, python_times, 'o-', label='Python/Numba', linewidth=2, markersize=8)
    ax1.plot(node_counts, rust_times, 's-', label='Rust (Single)', linewidth=2, markersize=8)
    ax1.plot(node_counts, rust_parallel_times, '^-', label='Rust (Parallel)', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Clustering Performance Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    
    # Plot 2: Speedup over Python
    ax2.plot(node_counts, speedups, 'o-', label='Rust vs Python', linewidth=2, markersize=8, color='green')
    ax2.plot(node_counts, parallel_speedups, 's-', label='Rust Parallel vs Python', linewidth=2, markersize=8, color='red')
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No speedup')
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Speedup vs Python Implementation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # Plot 3: Memory scaling (edges vs nodes)
    edge_counts = [r['num_edges'] for r in results]
    ax3.plot(node_counts, edge_counts, 'o-', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Nodes')
    ax3.set_ylabel('Number of Edges')
    ax3.set_title('Graph Size Scaling')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    
    # Plot 4: Cluster quality comparison
    python_clusters = [r.get('python_clusters', np.nan) for r in results]
    rust_clusters = [r.get('rust_clusters', np.nan) for r in results]
    target_clusters = [r['num_clusters'] for r in results]
    
    ax4.plot(node_counts, target_clusters, 'k--', label='Target', linewidth=2)
    ax4.plot(node_counts, python_clusters, 'o-', label='Python Result', linewidth=2, markersize=8)
    ax4.plot(node_counts, rust_clusters, 's-', label='Rust Result', linewidth=2, markersize=8)
    ax4.set_xlabel('Number of Nodes')
    ax4.set_ylabel('Number of Clusters')
    ax4.set_title('Clustering Quality Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(__file__).parent.parent / "benchmark_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Benchmark clustering algorithms")
    parser.add_argument("--nodes", type=int, help="Test specific number of nodes")
    parser.add_argument("--clusters", type=int, help="Number of clusters")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting results")
    
    args = parser.parse_args()
    
    print("L2G Clustering Benchmark")
    print("=" * 40)
    print(f"Rust available: {is_rust_available()}")
    
    if args.nodes and args.clusters:
        # Single test
        results = [benchmark_single_size(args.nodes, args.clusters, args.runs)]
    else:
        # Scaling benchmark
        results = run_scaling_benchmark()
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    for result in results:
        print(f"\nNodes: {result['num_nodes']}, Clusters: {result['num_clusters']}")
        if 'python_mean' in result:
            print(f"  Python:  {result['python_mean']:.3f}s ± {result['python_std']:.3f}s")
        if 'rust_mean' in result:
            print(f"  Rust:    {result['rust_mean']:.3f}s ± {result['rust_std']:.3f}s")
            if 'speedup' in result:
                print(f"  Speedup: {result['speedup']:.2f}x")
        if 'rust_parallel_mean' in result:
            print(f"  Rust||:  {result['rust_parallel_mean']:.3f}s ± {result['rust_parallel_std']:.3f}s")
            if 'parallel_speedup' in result:
                print(f"  Par||up: {result['parallel_speedup']:.2f}x")
    
    # Plot results
    if not args.no_plot and results:
        try:
            plot_results(results)
        except Exception as e:
            print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()