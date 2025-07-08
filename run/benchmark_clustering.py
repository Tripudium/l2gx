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
from pathlib import Path
import argparse
import sys

# Add L2G to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from l2gx.patch.clustering.fennel import fennel_clustering_safe
from l2gx.patch.clustering.rust_fennel import (
    fennel_clustering_rust, 
    is_rust_available
)
from test_graph_utils import generate_hidden_partition_model



def benchmark_single_size(num_nodes: int, num_clusters: int, num_runs: int = 3) -> dict:
    """Benchmark clustering algorithms on a single graph size"""
    print(f"\nBenchmarking {num_nodes} nodes, {num_clusters} clusters...")
    
    # Generate test graph
    graph, cluster_assignments = generate_hidden_partition_model(num_nodes, num_clusters, 0.8, 0.5)
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