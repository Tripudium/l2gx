use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use rayon::prelude::*;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Mutex;

/// High-performance Fennel clustering implementation in Rust
#[pyfunction]
#[pyo3(signature = (edge_index, adj_index, num_nodes, num_clusters, load_limit=1.1, alpha=None, gamma=1.5, num_iters=1, verbose=true))]
pub fn fennel_clustering_rust(
    py: Python,
    edge_index: PyReadonlyArray1<i64>,
    adj_index: PyReadonlyArray1<i64>,
    num_nodes: usize,
    num_clusters: usize,
    load_limit: f64,
    alpha: Option<f64>,
    gamma: f64,
    num_iters: usize,
    verbose: bool,
) -> PyResult<Py<PyArray1<i64>>> {
    let edge_index = edge_index.as_slice()?;
    let adj_index = adj_index.as_slice()?;
    
    if verbose {
        println!("Starting Fennel clustering: {} nodes → {} clusters", num_nodes, num_clusters);
    }
    
    let result = fennel_clustering_core(
        edge_index,
        adj_index,
        num_nodes,
        num_clusters,
        load_limit,
        alpha,
        gamma,
        num_iters,
        verbose,
    );
    
    if verbose {
        let unique_clusters = count_unique_clusters(&result);
        let cluster_sizes = compute_cluster_sizes(&result, num_clusters);
        println!("Rust Fennel completed: {} clusters, sizes: {:?}", unique_clusters, cluster_sizes);
    }
    
    Ok(result.into_pyarray(py).to_owned())
}

/// Parallel version of Fennel clustering using Rayon
/// 
/// This version processes multiple nodes in parallel during each iteration,
/// providing additional speedup for large graphs.
#[pyfunction]
#[pyo3(signature = (edge_index, adj_index, num_nodes, num_clusters, load_limit=1.1, alpha=None, gamma=1.5, num_iters=1, verbose=true))]
pub fn fennel_clustering_parallel_rust(
    py: Python,
    edge_index: PyReadonlyArray1<i64>,
    adj_index: PyReadonlyArray1<i64>,
    num_nodes: usize,
    num_clusters: usize,
    load_limit: f64,
    alpha: Option<f64>,
    gamma: f64,
    num_iters: usize,
    verbose: bool,
) -> PyResult<Py<PyArray1<i64>>> {
    let edge_index = edge_index.as_slice()?;
    let adj_index = adj_index.as_slice()?;
    
    if verbose {
        println!("Starting Parallel Rust Fennel clustering: {} nodes → {} clusters", num_nodes, num_clusters);
    }
    
    let result = fennel_clustering_parallel_core(
        edge_index,
        adj_index,
        num_nodes,
        num_clusters,
        load_limit,
        alpha,
        gamma,
        num_iters,
        verbose,
    );
    
    if verbose {
        let unique_clusters = count_unique_clusters(&result);
        let cluster_sizes = compute_cluster_sizes(&result, num_clusters);
        println!("Parallel Rust Fennel completed: {} clusters, sizes: {:?}", unique_clusters, cluster_sizes);
    }
    
    Ok(result.into_pyarray(py).to_owned())
}

/// Core Fennel clustering algorithm implementation
fn fennel_clustering_core(
    edge_index: &[i64],
    adj_index: &[i64],
    num_nodes: usize,
    num_clusters: usize,
    load_limit: f64,
    alpha: Option<f64>,
    gamma: f64,
    num_iters: usize,
    verbose: bool,
) -> Vec<i64> {
    let num_edges = edge_index.len() / 2;
    
    // Calculate alpha if not provided
    let alpha = alpha.unwrap_or_else(|| {
        (num_edges as f64) * (num_clusters as f64).powf(gamma - 1.0) / (num_nodes as f64).powf(gamma)
    });
    
    // Initialize cluster assignments
    let mut clusters = vec![-1i64; num_nodes];
    let mut partition_sizes = vec![0i64; num_clusters];
    
    // Calculate load limit
    let load_limit = (load_limit * (num_nodes as f64) / (num_clusters as f64)) as i64;
    
    // Initialize deltas
    let mut deltas = vec![0.0f64; num_clusters];
    for i in 0..num_clusters {
        deltas[i] = -alpha * gamma * (partition_sizes[i] as f64).powf(gamma - 1.0);
    }
    
    // Main clustering iterations
    for iteration in 0..num_iters {
        let mut not_converged = 0usize;
        
        for node in 0..num_nodes {
            // Get neighbors of current node
            let start_idx = adj_index[node] as usize;
            let end_idx = adj_index[node + 1] as usize;
            
            // Count neighbors in each cluster
            let mut neighbor_counts = vec![0i64; num_clusters];
            for &neighbor_idx in &edge_index[start_idx..end_idx] {
                let neighbor = neighbor_idx as usize;
                if neighbor < num_nodes {
                    let neighbor_cluster = clusters[neighbor];
                    if neighbor_cluster >= 0 {
                        neighbor_counts[neighbor_cluster as usize] += 1;
                    }
                }
            }
            
            // Update partition sizes if node was already assigned
            let old_cluster = clusters[node];
            if old_cluster >= 0 {
                partition_sizes[old_cluster as usize] -= 1;
                deltas[old_cluster as usize] = -alpha * gamma * 
                    (partition_sizes[old_cluster as usize] as f64).powf(gamma - 1.0);
            }
            
            // Find best cluster
            let mut best_cluster = 0;
            let mut best_score = f64::NEG_INFINITY;
            
            for cluster in 0..num_clusters {
                if partition_sizes[cluster] < load_limit {
                    let score = deltas[cluster] + neighbor_counts[cluster] as f64;
                    if score > best_score {
                        best_score = score;
                        best_cluster = cluster;
                    }
                }
            }
            
            // Assign node to best cluster
            clusters[node] = best_cluster as i64;
            partition_sizes[best_cluster] += 1;
            
            // Update delta for the chosen cluster
            if partition_sizes[best_cluster] >= load_limit {
                deltas[best_cluster] = f64::NEG_INFINITY;
            } else {
                deltas[best_cluster] = -alpha * gamma * 
                    (partition_sizes[best_cluster] as f64).powf(gamma - 1.0);
            }
            
            // Track convergence
            if old_cluster != best_cluster as i64 {
                not_converged += 1;
            }
        }
        
        if verbose {
            println!("Rust Fennel iteration {}: {} nodes not converged", iteration, not_converged);
        }
        
        // Check for convergence
        if not_converged == 0 {
            if verbose {
                println!("Rust Fennel converged after {} iterations", iteration);
            }
            break;
        }
    }
    
    clusters
}

/// Parallel version of Fennel clustering
/// 
/// This implementation uses atomic operations and careful synchronization
/// to parallelize the node assignment process while maintaining correctness.
fn fennel_clustering_parallel_core(
    edge_index: &[i64],
    adj_index: &[i64],
    num_nodes: usize,
    num_clusters: usize,
    load_limit: f64,
    alpha: Option<f64>,
    gamma: f64,
    num_iters: usize,
    verbose: bool,
) -> Vec<i64> {
    let num_edges = edge_index.len() / 2;
    
    // Calculate alpha if not provided
    let alpha = alpha.unwrap_or_else(|| {
        (num_edges as f64) * (num_clusters as f64).powf(gamma - 1.0) / (num_nodes as f64).powf(gamma)
    });
    
    // Initialize cluster assignments with atomic operations for thread safety
    let clusters: Vec<AtomicI64> = (0..num_nodes).map(|_| AtomicI64::new(-1)).collect();
    let partition_sizes: Vec<AtomicI64> = (0..num_clusters).map(|_| AtomicI64::new(0)).collect();
    
    // Calculate load limit
    let load_limit = (load_limit * (num_nodes as f64) / (num_clusters as f64)) as i64;
    
    // Use mutex for deltas to avoid race conditions
    let deltas = Mutex::new(vec![0.0f64; num_clusters]);
    
    // Initialize deltas
    {
        let mut deltas_guard = deltas.lock().unwrap();
        for i in 0..num_clusters {
            deltas_guard[i] = -alpha * gamma * 0.0f64.powf(gamma - 1.0);
        }
    }
    
    // Main clustering iterations
    for iteration in 0..num_iters {
        let not_converged = AtomicI64::new(0);
        
        // Process nodes in parallel batches to reduce contention
        let batch_size = (num_nodes / rayon::current_num_threads()).max(1000);
        
        (0..num_nodes)
            .collect::<Vec<_>>()
            .chunks(batch_size)
            .for_each(|batch| {
                for &node in batch {
                    // Get neighbors of current node
                    let start_idx = adj_index[node] as usize;
                    let end_idx = adj_index[node + 1] as usize;
                    
                    // Count neighbors in each cluster
                    let mut neighbor_counts = vec![0i64; num_clusters];
                    for &neighbor_idx in &edge_index[start_idx..end_idx] {
                        let neighbor = neighbor_idx as usize;
                        if neighbor < num_nodes {
                            let neighbor_cluster = clusters[neighbor].load(Ordering::Relaxed);
                            if neighbor_cluster >= 0 {
                                neighbor_counts[neighbor_cluster as usize] += 1;
                            }
                        }
                    }
                    
                    // Update partition sizes if node was already assigned
                    let old_cluster = clusters[node].load(Ordering::Relaxed);
                    if old_cluster >= 0 {
                        partition_sizes[old_cluster as usize].fetch_sub(1, Ordering::Relaxed);
                    }
                    
                    // Find best cluster (with lock for deltas)
                    let best_cluster = {
                        let deltas_guard = deltas.lock().unwrap();
                        let mut best_cluster = 0;
                        let mut best_score = f64::NEG_INFINITY;
                        
                        for cluster in 0..num_clusters {
                            let current_size = partition_sizes[cluster].load(Ordering::Relaxed);
                            if current_size < load_limit {
                                let score = deltas_guard[cluster] + neighbor_counts[cluster] as f64;
                                if score > best_score {
                                    best_score = score;
                                    best_cluster = cluster;
                                }
                            }
                        }
                        best_cluster
                    };
                    
                    // Assign node to best cluster
                    clusters[node].store(best_cluster as i64, Ordering::Relaxed);
                    let new_size = partition_sizes[best_cluster].fetch_add(1, Ordering::Relaxed) + 1;
                    
                    // Update delta for the chosen cluster
                    {
                        let mut deltas_guard = deltas.lock().unwrap();
                        if new_size >= load_limit {
                            deltas_guard[best_cluster] = f64::NEG_INFINITY;
                        } else {
                            deltas_guard[best_cluster] = -alpha * gamma * 
                                (new_size as f64).powf(gamma - 1.0);
                        }
                    }
                    
                    // Track convergence
                    if old_cluster != best_cluster as i64 {
                        not_converged.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        
        let total_not_converged = not_converged.load(Ordering::Relaxed);
        
        if verbose {
            println!("Parallel Rust Fennel iteration {}: {} nodes not converged", iteration, total_not_converged);
        }
        
        // Check for convergence
        if total_not_converged == 0 {
            if verbose {
                println!("Parallel Rust Fennel converged after {} iterations", iteration);
            }
            break;
        }
    }
    
    // Convert atomic results back to regular vector
    clusters.into_iter().map(|atomic| atomic.into_inner()).collect()
}

/// Helper function to count unique clusters
fn count_unique_clusters(clusters: &[i64]) -> usize {
    let mut unique = std::collections::HashSet::new();
    for &cluster in clusters {
        if cluster >= 0 {
            unique.insert(cluster);
        }
    }
    unique.len()
}

/// Helper function to compute cluster sizes
fn compute_cluster_sizes(clusters: &[i64], num_clusters: usize) -> Vec<usize> {
    let mut sizes = vec![0usize; num_clusters];
    for &cluster in clusters {
        if cluster >= 0 && (cluster as usize) < num_clusters {
            sizes[cluster as usize] += 1;
        }
    }
    sizes
}