#!/usr/bin/env python3
"""
Quick test to check if methods are predicting only majority class.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from l2gx.datasets import get_dataset
from l2gx.embedding import get_embedding
from l2gx.align import get_aligner


def test_classification(embeddings, labels, method_name, label_names):
    """Test if classification is working properly."""
    
    print(f"\n{'='*50}")
    print(f"{method_name}")
    print(f"{'='*50}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    clf = LogisticRegression(
        max_iter=2000,
        solver='lbfgs',
        multi_class='ovr',
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_scaled)
    
    # Check predictions
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    
    unique_pred = np.unique(y_pred)
    unique_true = np.unique(y_test)
    
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Unique true classes: {len(unique_true)}")
    print(f"Unique predicted classes: {len(unique_pred)}")
    
    if len(unique_pred) == 1:
        print(f"⚠️ ONLY PREDICTING CLASS {unique_pred[0]} ({label_names[unique_pred[0]]})")
    
    # Show distribution
    from collections import Counter
    pred_counts = Counter(y_pred)
    
    print("\nPrediction distribution:")
    for label in sorted(pred_counts.keys()):
        count = pred_counts[label]
        pct = count / len(y_test) * 100
        print(f"  {label_names[label]:15s}: {count:3d} ({pct:5.1f}%)")
    
    # Check if it's majority class
    majority_class = Counter(y_test).most_common(1)[0][0]
    majority_name = label_names[majority_class]
    majority_accuracy = np.mean(y_test == majority_class)
    
    print(f"\nMajority class: {majority_name} ({majority_accuracy:.2%} of test set)")
    
    if abs(accuracy - majority_accuracy) < 0.01:
        print("⚠️ Accuracy matches majority class baseline!")
    
    return accuracy, len(unique_pred)


def main():
    # Load data
    print("Loading BTC-reduced dataset...")
    dataset = get_dataset("btc-reduced", max_nodes=2000)
    data = dataset[0]
    labels = data.y.numpy()
    label_names = data.label_names
    
    print(f"Dataset: {data.num_nodes} nodes")
    
    # Test each method with dim=128
    dim = 128
    
    # 1. VGAE (baseline)
    print("\nGenerating VGAE embeddings...")
    vgae = get_embedding("vgae", embedding_dim=dim, epochs=50, verbose=False)
    vgae_emb = vgae.fit_transform(data)
    test_classification(vgae_emb, labels, "VGAE (Baseline)", label_names)
    
    # 2. GraphSAGE
    print("\nGenerating GraphSAGE embeddings...")
    sage = get_embedding("graphsage", embedding_dim=dim, epochs=100, 
                        learning_rate=0.01, verbose=False)
    sage_emb = sage.fit_transform(data)
    test_classification(sage_emb, labels, "GraphSAGE", label_names)
    
    # 3. Hierarchical
    print("\nGenerating Hierarchical embeddings...")
    aligner = get_aligner("l2g")
    hier = get_embedding("hierarchical", embedding_dim=dim, aligner=aligner,
                        max_patch_size=800, base_method="vgae", epochs=50, 
                        verbose=False)
    hier_emb = hier.fit_transform(data)
    test_classification(hier_emb, labels, "Hierarchical", label_names)


if __name__ == "__main__":
    main()