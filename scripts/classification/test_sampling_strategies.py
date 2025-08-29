#!/usr/bin/env python3
"""
Quick test to demonstrate the effectiveness of different sampling strategies
for handling class imbalance in BTC-reduced dataset.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score,
    classification_report,
    f1_score
)
from collections import Counter

# Import sampling methods
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

from l2gx.datasets import get_dataset
from l2gx.embedding import get_embedding


def test_sampling_strategy(X_train, X_test, y_train, y_test, strategy_name, sampler=None):
    """Test a single sampling strategy."""
    print(f"\n{'='*50}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*50}")
    
    # Apply sampling if provided
    if sampler is not None:
        try:
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            print(f"Resampled: {len(y_train)} → {len(y_resampled)} samples")
            
            # Show new distribution
            counter = Counter(y_resampled)
            print("New distribution:", dict(sorted(counter.items())))
        except Exception as e:
            print(f"Sampling failed: {e}")
            X_resampled, y_resampled = X_train, y_train
    else:
        X_resampled, y_resampled = X_train, y_train
    
    # Train classifier
    if strategy_name == "Class Weights":
        clf = LogisticRegression(
            max_iter=2000, 
            solver='lbfgs', 
            multi_class='ovr',
            class_weight='balanced',  # Use balanced class weights
            random_state=42
        )
    else:
        clf = LogisticRegression(
            max_iter=2000, 
            solver='lbfgs', 
            multi_class='ovr',
            random_state=42
        )
    
    clf.fit(X_resampled, y_resampled)
    y_pred = clf.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\nMetrics:")
    print(f"  Accuracy:          {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Balanced Accuracy: {bal_acc:.4f} ({bal_acc*100:.2f}%)")
    print(f"  F1 Macro:          {f1_macro:.4f}")
    print(f"  F1 Weighted:       {f1_weighted:.4f}")
    
    # Check prediction diversity
    unique_preds = len(np.unique(y_pred))
    print(f"  Unique predictions: {unique_preds} classes")
    
    # Show per-class performance for minority classes
    if hasattr(y_test, '__len__'):
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0
        )
        
        # Find minority classes (< 5% of data)
        total = len(y_test)
        minority_classes = [i for i, s in enumerate(support) if s < total * 0.05]
        
        if minority_classes:
            print(f"\nMinority class performance:")
            for cls in minority_classes[:5]:  # Show top 5 minority classes
                print(f"  Class {cls}: F1={f1[cls]:.3f}, Precision={precision[cls]:.3f}, "
                      f"Recall={recall[cls]:.3f} (n={support[cls]})")
    
    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


def main():
    print("Testing Sampling Strategies for BTC-Reduced Class Imbalance")
    print("="*60)
    
    # Load dataset
    print("\nLoading BTC-reduced dataset...")
    dataset = get_dataset("btc-reduced", max_nodes=3000)
    data = dataset[0]
    
    print(f"Dataset: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    print(f"Classes: {data.num_classes}")
    
    # Get labels and show distribution
    labels = data.y.numpy()
    label_names = data.label_names
    
    print("\nOriginal class distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        pct = count / len(labels) * 100
        print(f"  {label_names[label]:15s}: {count:4d} ({pct:5.1f}%)")
    
    # Generate embeddings
    print("\nGenerating VGAE embeddings (dim=64)...")
    embedder = get_embedding("vgae", embedding_dim=64, epochs=50, verbose=False)
    embeddings = embedder.fit_transform(data)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")
    
    # Test different strategies
    results = {}
    
    # 1. Baseline (no sampling)
    results['Baseline'] = test_sampling_strategy(
        X_train_scaled, X_test_scaled, y_train, y_test,
        "Baseline (No Sampling)", None
    )
    
    # 2. Class weights
    results['Class Weights'] = test_sampling_strategy(
        X_train_scaled, X_test_scaled, y_train, y_test,
        "Class Weights", None
    )
    
    # 3. Random Over-Sampling
    results['ROS'] = test_sampling_strategy(
        X_train_scaled, X_test_scaled, y_train, y_test,
        "Random Over-Sampling (ROS)",
        RandomOverSampler(random_state=42)
    )
    
    # 4. SMOTE
    # Adjust k_neighbors for small classes
    min_class_size = min(Counter(y_train).values())
    k_neighbors = min(5, min_class_size - 1)
    k_neighbors = max(1, k_neighbors)
    
    results['SMOTE'] = test_sampling_strategy(
        X_train_scaled, X_test_scaled, y_train, y_test,
        f"SMOTE (k={k_neighbors})",
        SMOTE(k_neighbors=k_neighbors, random_state=42)
    )
    
    # 5. Random Under-Sampling
    results['RUS'] = test_sampling_strategy(
        X_train_scaled, X_test_scaled, y_train, y_test,
        "Random Under-Sampling (RUS)",
        RandomUnderSampler(random_state=42)
    )
    
    # 6. Combined SMOTE + ENN
    results['SMOTE+ENN'] = test_sampling_strategy(
        X_train_scaled, X_test_scaled, y_train, y_test,
        "SMOTE + Edited Nearest Neighbors",
        SMOTEENN(random_state=42, smote=SMOTE(k_neighbors=k_neighbors))
    )
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    
    print(f"\n{'Strategy':<20} {'Accuracy':<10} {'Balanced':<10} {'F1 Macro':<10} {'F1 Weight':<10}")
    print("-"*60)
    
    for strategy, metrics in results.items():
        print(f"{strategy:<20} {metrics['accuracy']:.4f}     "
              f"{metrics['balanced_accuracy']:.4f}     "
              f"{metrics['f1_macro']:.4f}     "
              f"{metrics['f1_weighted']:.4f}")
    
    # Find best strategy
    best_balanced = max(results.items(), key=lambda x: x[1]['balanced_accuracy'])
    best_f1 = max(results.items(), key=lambda x: x[1]['f1_macro'])
    
    print(f"\n🏆 Best Balanced Accuracy: {best_balanced[0]} ({best_balanced[1]['balanced_accuracy']:.4f})")
    print(f"🏆 Best F1 Macro Score:    {best_f1[0]} ({best_f1[1]['f1_macro']:.4f})")
    
    # Improvement analysis
    baseline_bal = results['Baseline']['balanced_accuracy']
    baseline_f1 = results['Baseline']['f1_macro']
    
    print(f"\nImprovement over baseline:")
    for strategy, metrics in results.items():
        if strategy != 'Baseline':
            bal_improve = (metrics['balanced_accuracy'] - baseline_bal) / baseline_bal * 100
            f1_improve = (metrics['f1_macro'] - baseline_f1) / baseline_f1 * 100
            print(f"  {strategy:<20}: {bal_improve:+.1f}% bal_acc, {f1_improve:+.1f}% F1")


if __name__ == "__main__":
    main()