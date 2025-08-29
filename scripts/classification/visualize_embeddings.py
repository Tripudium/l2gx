#!/usr/bin/env python3
"""
Visualize embeddings using UMAP to diagnose classification issues.
Generates embeddings for different methods and creates UMAP visualizations.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import umap
import torch

from l2gx.datasets import get_dataset
from l2gx.embedding import get_embedding
from l2gx.align import get_aligner
from l2gx.graphs import TGraph


def generate_embeddings(data, method_name, embedding_dim=128, verbose=True):
    """Generate embeddings for a specific method."""
    
    if verbose:
        print(f"\nGenerating {method_name} embeddings (dim={embedding_dim})...")
    
    try:
        if method_name == "graphsage_full":
            embedder = get_embedding(
                "graphsage",
                embedding_dim=embedding_dim,
                hidden_dim=embedding_dim * 2,
                epochs=200,
                learning_rate=0.01,
                patience=30,
                verbose=verbose
            )
            embeddings = embedder.fit_transform(data)
            
        elif method_name == "hierarchical_unified":
            aligner = get_aligner("l2g")
            embedder = get_embedding(
                "hierarchical",
                embedding_dim=embedding_dim,
                aligner=aligner,
                max_patch_size=800,
                base_method="vgae",
                min_overlap=64,
                target_overlap=128,
                epochs=100,
                learning_rate=0.001,
                patience=20,
                verbose=verbose
            )
            embeddings = embedder.fit_transform(data)
            
        elif method_name == "vgae_baseline":
            embedder = get_embedding(
                "vgae",
                embedding_dim=embedding_dim,
                hidden_dim=embedding_dim * 2,
                epochs=100,
                learning_rate=0.001,
                patience=20,
                verbose=verbose
            )
            embeddings = embedder.fit_transform(data)
            
        else:
            raise ValueError(f"Unknown method: {method_name}")
            
        if verbose:
            print(f"  Generated embeddings: {embeddings.shape}")
            print(f"  Mean: {np.mean(embeddings):.4f}, Std: {np.std(embeddings):.4f}")
            print(f"  Min: {np.min(embeddings):.4f}, Max: {np.max(embeddings):.4f}")
            
        return embeddings
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_with_umap(embeddings, labels, method_name, label_names=None, save_path=None):
    """Create UMAP visualization of embeddings."""
    
    print(f"\nCreating UMAP visualization for {method_name}...")
    
    # Check for degenerate embeddings
    if np.std(embeddings) < 1e-6:
        print(f"  WARNING: Embeddings have very low variance (std={np.std(embeddings):.6f})")
        print(f"  This suggests the embedding method may have collapsed!")
    
    # Create UMAP projection
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42,
        metric='euclidean'
    )
    
    try:
        umap_embeddings = reducer.fit_transform(embeddings)
    except Exception as e:
        print(f"  ERROR in UMAP: {e}")
        # If UMAP fails, try with PCA fallback
        from sklearn.decomposition import PCA
        print("  Falling back to PCA...")
        pca = PCA(n_components=2)
        umap_embeddings = pca.fit_transform(embeddings)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: colored by labels
    ax1 = axes[0]
    
    # Get unique labels and create color map
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = label_names[label] if label_names else f"Class {label}"
        ax1.scatter(
            umap_embeddings[mask, 0],
            umap_embeddings[mask, 1],
            c=[colors[i]],
            label=label_name,
            alpha=0.6,
            s=10
        )
    
    ax1.set_title(f"{method_name} - UMAP Projection (Colored by True Labels)")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Density/overlap visualization
    ax2 = axes[1]
    
    # Create hexbin plot to show density
    hexbin = ax2.hexbin(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        gridsize=30,
        cmap='YlOrRd',
        mincnt=1
    )
    
    ax2.set_title(f"{method_name} - Density Plot")
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    plt.colorbar(hexbin, ax=ax2, label='Point Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to {save_path}")
    
    plt.show()
    
    return umap_embeddings


def analyze_embeddings(embeddings, labels, method_name, label_names=None):
    """Analyze embedding quality and classification performance."""
    
    print(f"\n{'='*60}")
    print(f"Analysis for {method_name}")
    print(f"{'='*60}")
    
    # 1. Check embedding statistics
    print("\n1. Embedding Statistics:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Mean: {np.mean(embeddings):.6f}")
    print(f"   Std: {np.std(embeddings):.6f}")
    print(f"   Min: {np.min(embeddings):.6f}")
    print(f"   Max: {np.max(embeddings):.6f}")
    
    # Check for collapsed embeddings
    col_std = np.std(embeddings, axis=0)
    zero_variance_dims = np.sum(col_std < 1e-6)
    print(f"   Dimensions with near-zero variance: {zero_variance_dims}/{embeddings.shape[1]}")
    
    if zero_variance_dims > embeddings.shape[1] * 0.5:
        print("   ⚠️ WARNING: Many dimensions have collapsed to constant values!")
    
    # 2. Check pairwise distances
    print("\n2. Pairwise Distance Analysis:")
    # Sample for efficiency
    n_sample = min(1000, len(embeddings))
    sample_idx = np.random.choice(len(embeddings), n_sample, replace=False)
    sample_emb = embeddings[sample_idx]
    
    from scipy.spatial.distance import pdist
    distances = pdist(sample_emb, metric='euclidean')
    
    print(f"   Mean distance: {np.mean(distances):.6f}")
    print(f"   Std distance: {np.std(distances):.6f}")
    print(f"   Min distance: {np.min(distances):.6f}")
    print(f"   Max distance: {np.max(distances):.6f}")
    
    if np.std(distances) < 0.01:
        print("   ⚠️ WARNING: Very low variation in pairwise distances - embeddings may be collapsed!")
    
    # 3. Classification test
    print("\n3. Classification Performance:")
    
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    classifier = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        multi_class='ovr',
        class_weight='balanced',
        random_state=42
    )
    
    classifier.fit(X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Check if it's predicting only one class
    unique_predictions = np.unique(y_pred)
    print(f"   Unique predictions: {len(unique_predictions)} classes")
    
    if len(unique_predictions) == 1:
        print(f"   ⚠️ WARNING: Classifier is only predicting class {unique_predictions[0]}!")
        if label_names:
            print(f"   (Class name: {label_names[unique_predictions[0]]})")
    
    # Show prediction distribution
    print("\n4. Prediction Distribution:")
    from collections import Counter
    pred_counts = Counter(y_pred)
    true_counts = Counter(y_test)
    
    print("   True distribution:")
    for label in sorted(true_counts.keys()):
        name = label_names[label] if label_names else f"Class {label}"
        count = true_counts[label]
        pct = count / len(y_test) * 100
        print(f"     {name:15s}: {count:4d} ({pct:5.1f}%)")
    
    print("\n   Predicted distribution:")
    for label in sorted(pred_counts.keys()):
        name = label_names[label] if label_names else f"Class {label}"
        count = pred_counts[label]
        pct = count / len(y_test) * 100
        print(f"     {name:15s}: {count:4d} ({pct:5.1f}%)")
    
    return accuracy, y_pred, y_test


def main():
    """Main function to run embedding visualization and analysis."""
    
    print("="*60)
    print("Embedding Visualization and Diagnosis")
    print("="*60)
    
    # Load BTC-reduced dataset
    print("\nLoading BTC-reduced dataset...")
    dataset = get_dataset("btc-reduced", max_nodes=3000)  # Moderate size for testing
    data = dataset[0]
    
    print(f"Dataset: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    print(f"Features: {data.x.shape}, Labels: {data.y.shape}")
    print(f"Classes: {data.num_classes}")
    
    # Get labels and label names
    labels = data.y.numpy()
    label_names = data.label_names
    
    # Methods to test
    methods = [
        "vgae_baseline",       # Known to work
        "graphsage_full",      # Suspected issue
        "hierarchical_unified" # Suspected issue
    ]
    
    results = {}
    
    # Generate and analyze embeddings for each method
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Testing {method}")
        print(f"{'='*60}")
        
        # Generate embeddings
        embeddings = generate_embeddings(data, method, embedding_dim=128, verbose=True)
        
        if embeddings is None:
            print(f"Skipping {method} due to error")
            continue
        
        # Analyze embeddings
        accuracy, y_pred, y_test = analyze_embeddings(
            embeddings, labels, method, label_names
        )
        
        # Visualize with UMAP
        save_path = f"umap_{method}.png"
        umap_emb = visualize_with_umap(
            embeddings, labels, method, label_names, save_path
        )
        
        results[method] = {
            'embeddings': embeddings,
            'umap': umap_emb,
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    
    print("\nAccuracy Comparison:")
    for method, res in results.items():
        if 'accuracy' in res:
            print(f"  {method:25s}: {res['accuracy']:.4f} ({res['accuracy']*100:.2f}%)")
    
    print("\nEmbedding Quality:")
    for method, res in results.items():
        if 'embeddings' in res:
            emb = res['embeddings']
            print(f"  {method:25s}: std={np.std(emb):.6f}, "
                  f"unique_pred={len(np.unique(res.get('predictions', [])))}")
    
    print("\n✅ Analysis complete! Check the generated UMAP visualizations:")
    for method in methods:
        print(f"  - umap_{method}.png")


if __name__ == "__main__":
    main()