#!/usr/bin/env python3
"""
PubMed Embedding Comparison: Patched vs Whole Graph

This script compares patched embedding (20 patches) vs whole graph embedding
of the PubMed dataset using UMAP visualization, with early stopping.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import umap
import seaborn as sns
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from l2gx.datasets import get_dataset
from l2gx.embedding import get_embedding

def setup_plot_style():
    """Set up matplotlib style for clean plots."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams.update({
        'figure.figsize': (16, 8),
        'font.size': 12,
        'legend.fontsize': 11,
        'xtick.labelsize': 0,  # Hide tick labels
        'ytick.labelsize': 0,  # Hide tick labels
        'axes.labelsize': 0,   # Hide axis labels
        'axes.titlesize': 0,   # Hide titles
    })

def load_pubmed_data():
    """Load and analyze the PubMed dataset."""
    
    print("📚 LOADING PUBMED DATASET")
    print("=" * 50)
    
    dataset = get_dataset("PubMed")
    data = dataset.to("torch-geometric")
    labels = data.y.cpu().numpy()
    
    print(f"Dataset loaded successfully!")
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.num_edges:,}")
    print(f"  Features: {data.x.shape[1]} dimensions")
    print(f"  Classes: {data.y.max().item() + 1}")
    print(f"  Average degree: {(2 * data.num_edges / data.num_nodes):.1f}")
    
    # Class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(labels)) * 100
        print(f"  Class {label}: {count:,} nodes ({percentage:.1f}%)")
    
    return data, labels

def compute_patched_embedding(data):
    """Compute patched embedding using the same parameters as the experiment."""
    
    print(f"\\n🔧 COMPUTING PATCHED EMBEDDING")
    print("=" * 50)
    
    # Use the same parameters as in the classification experiment
    print("Parameters:")
    print("  - Method: Patched VGAE")
    print("  - Number of patches: 20")
    print("  - Min overlap: 256")
    print("  - Target overlap: 512")
    print("  - Target patch degree: 5")
    print("  - Embedding dimension: 64")
    print("  - Epochs: 300 (increased)")
    print("  - Early stopping patience: 20")
    
    start_time = time.time()
    
    embedder = get_embedding(
        'patched',
        embedding_method='vgae',
        embedding_dim=128,
        num_patches=20,
        min_overlap=512,
        target_overlap=1024,
        target_patch_degree=5,
        clustering_method='metis',
        alignment_method='l2g',
        enable_scaling=True,
        epochs=1000,  # Increased epochs
        lr=0.01,
        patience=20,  # Early stopping patience
        verbose=True
    )
    
    print("\\nFitting patched embedder...")
    embeddings = embedder.fit_transform(data)
    
    embedding_time = time.time() - start_time
    
    print(f"\\n✅ Patched embedding complete!")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Computation time: {embedding_time:.1f} seconds")
    print(f"  Embedding statistics:")
    print(f"    Mean: {embeddings.mean():.4f}")
    print(f"    Std: {embeddings.std():.4f}")
    print(f"    Min: {embeddings.min():.4f}")
    print(f"    Max: {embeddings.max():.4f}")
    
    # Get patch information
    patches = embedder.get_patches()
    if patches:
        patch_sizes = [len(patch.nodes) for patch in patches]
        print(f"  Patch statistics:")
        print(f"    Number of patches: {len(patches)}")
        print(f"    Patch sizes: min={min(patch_sizes)}, max={max(patch_sizes)}, avg={np.mean(patch_sizes):.1f}")
    
    return embeddings, embedder

def compute_whole_graph_embedding(data):
    """Compute whole graph embedding for comparison."""
    
    print(f"\\n🔧 COMPUTING WHOLE GRAPH EMBEDDING")
    print("=" * 50)
    
    print("Parameters:")
    print("  - Method: VGAE (Whole Graph)")
    print("  - Embedding dimension: 64")
    print("  - Epochs: 300 (increased)")
    print("  - Early stopping patience: 20")
    
    start_time = time.time()
    
    embedder = get_embedding(
        'vgae',
        embedding_dim=128,
        epochs=1000,  # Increased epochs
        lr=0.01,
        patience=20,  # Early stopping patience
        verbose=True
    )
    
    print("\\nFitting whole graph embedder...")
    embeddings = embedder.fit_transform(data)
    
    embedding_time = time.time() - start_time
    
    print(f"\\n✅ Whole graph embedding complete!")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Computation time: {embedding_time:.1f} seconds")
    print(f"  Embedding statistics:")
    print(f"    Mean: {embeddings.mean():.4f}")
    print(f"    Std: {embeddings.std():.4f}")
    print(f"    Min: {embeddings.min():.4f}")
    print(f"    Max: {embeddings.max():.4f}")
    
    return embeddings, embedder

def create_visualizations(patched_embeddings, whole_embeddings, labels):
    """Create UMAP visualizations comparing patched vs whole graph embeddings."""
    
    print(f"\\n🎨 CREATING UMAP VISUALIZATIONS")
    print("=" * 50)
    
    setup_plot_style()
    
    # Define class names for PubMed (if known, otherwise use generic labels)
    class_names = ['Diabetes Mellitus Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
    colors = sns.color_palette("husl", len(class_names))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Compute UMAP for patched embeddings
    print("Computing UMAP projection for patched embeddings...")
    umap_reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        n_jobs=1  # Use single job to avoid potential issues
    )
    patched_umap = umap_reducer.fit_transform(patched_embeddings)
    
    # Plot patched UMAP
    for class_idx in range(len(class_names)):
        mask = labels == class_idx
        if np.any(mask):
            ax1.scatter(
                patched_umap[mask, 0], 
                patched_umap[mask, 1],
                c=[colors[class_idx]], 
                #label=class_names[class_idx],
                alpha=0.7,
                s=15,
                edgecolors='white',
                linewidth=0.1
            )
    
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Compute UMAP for whole graph embeddings
    print("Computing UMAP projection for whole graph embeddings...")
    umap_reducer2 = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        n_jobs=1
    )
    whole_umap = umap_reducer2.fit_transform(whole_embeddings)
    
    # Plot whole graph UMAP
    for class_idx in range(len(class_names)):
        mask = labels == class_idx
        if np.any(mask):
            ax2.scatter(
                whole_umap[mask, 0], 
                whole_umap[mask, 1],
                c=[colors[class_idx]], 
                label=class_names[class_idx],
                alpha=0.7,
                s=15,
                edgecolors='white',
                linewidth=0.1
            )
    
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('pubmed_embedding_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'pubmed_embedding_comparison.png'")
    
    # Show the plot
    plt.show()
    
    return patched_umap, whole_umap

def analyze_clustering_quality(patched_embeddings, whole_embeddings, labels, patched_umap, whole_umap):
    """Analyze the clustering quality of both embedding methods."""
    
    print(f"\\n📊 CLUSTERING QUALITY ANALYSIS")
    print("=" * 50)
    
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.cluster import KMeans
    
    n_classes = len(np.unique(labels))
    
    # Silhouette scores for patched method
    sil_patched_orig = silhouette_score(patched_embeddings, labels)
    sil_patched_umap = silhouette_score(patched_umap, labels)
    
    # Silhouette scores for whole graph method
    sil_whole_orig = silhouette_score(whole_embeddings, labels)
    sil_whole_umap = silhouette_score(whole_umap, labels)
    
    print(f"Silhouette Scores:")
    print(f"  Patched embedding (64D):    {sil_patched_orig:.4f}")
    print(f"  Patched UMAP projection:     {sil_patched_umap:.4f}")
    print(f"  Whole graph embedding (64D): {sil_whole_orig:.4f}")
    print(f"  Whole graph UMAP projection: {sil_whole_umap:.4f}")
    
    # K-means clustering evaluation
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    
    # Patched embeddings
    cluster_labels_patched = kmeans.fit_predict(patched_embeddings)
    ari_patched = adjusted_rand_score(labels, cluster_labels_patched)
    
    cluster_labels_patched_umap = kmeans.fit_predict(patched_umap)
    ari_patched_umap = adjusted_rand_score(labels, cluster_labels_patched_umap)
    
    # Whole graph embeddings
    cluster_labels_whole = kmeans.fit_predict(whole_embeddings)
    ari_whole = adjusted_rand_score(labels, cluster_labels_whole)
    
    cluster_labels_whole_umap = kmeans.fit_predict(whole_umap)
    ari_whole_umap = adjusted_rand_score(labels, cluster_labels_whole_umap)
    
    print(f"\\nAdjusted Rand Index (K-means clustering):")
    print(f"  Patched embedding (64D):    {ari_patched:.4f}")
    print(f"  Patched UMAP projection:     {ari_patched_umap:.4f}")
    print(f"  Whole graph embedding (64D): {ari_whole:.4f}")
    print(f"  Whole graph UMAP projection: {ari_whole_umap:.4f}")
    
    # Class separation analysis
    print(f"\\nClass Separation Analysis:")
    print(f"  Number of classes: {n_classes}")
    for i, class_name in enumerate(['Class 0', 'Class 1', 'Class 2']):
        count = np.sum(labels == i)
        percentage = (count / len(labels)) * 100
        print(f"  {class_name}: {count:,} nodes ({percentage:.1f}%)")
    
    # Comparison summary
    print(f"\\nComparison Summary:")
    print(f"  Better silhouette (original): {'Whole Graph' if sil_whole_orig > sil_patched_orig else 'Patched'}")
    print(f"  Better silhouette (UMAP):     {'Whole Graph' if sil_whole_umap > sil_patched_umap else 'Patched'}")
    print(f"  Better clustering (original):  {'Whole Graph' if ari_whole > ari_patched else 'Patched'}")
    print(f"  Better clustering (UMAP):      {'Whole Graph' if ari_whole_umap > ari_patched_umap else 'Patched'}")

def main():
    """Main function to run the embedding comparison visualization."""
    
    print("🔬 PUBMED EMBEDDING COMPARISON: PATCHED vs WHOLE GRAPH")
    print("=" * 80)
    print("This script compares patched vs whole graph embeddings of the PubMed dataset")
    print("using UMAP visualization with increased epochs and early stopping.")
    
    # Load dataset
    data, labels = load_pubmed_data()
    
    # Compute both embeddings
    patched_embeddings, patched_embedder = compute_patched_embedding(data)
    whole_embeddings, whole_embedder = compute_whole_graph_embedding(data)
    
    # Create visualizations
    patched_umap, whole_umap = create_visualizations(patched_embeddings, whole_embeddings, labels)
    
    # Analyze clustering quality
    analyze_clustering_quality(patched_embeddings, whole_embeddings, labels, patched_umap, whole_umap)
    
    print(f"\\n✅ COMPARISON COMPLETE!")
    print("=" * 50)
    print("Generated files:")
    print("• pubmed_embedding_comparison.png - UMAP comparison visualization")
    print("\\nThe visualization compares patched VGAE (left) vs whole graph VGAE (right)")
    print("embeddings of the PubMed dataset using UMAP projections.")
    print("Each point represents a document, colored by its diabetes research category.")

if __name__ == "__main__":
    main()