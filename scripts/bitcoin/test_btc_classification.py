#!/usr/bin/env python3
"""
Test BTC Dataset Classification Pipeline

Verifies that the BTC-reduced dataset works with the classification framework
by running a quick classification test with a simple VGAE embedding.
"""

import sys
from pathlib import Path

# Add l2gx to path
sys.path.insert(0, str(Path(__file__).parent))

from l2gx.datasets import get_dataset
from l2gx.embedding import get_embedding
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def test_btc_classification():
    """Test basic classification pipeline with BTC dataset."""
    print("Testing BTC Dataset Classification Pipeline")
    print("=" * 60)
    
    # Load BTC reduced dataset
    print("1. Loading BTC reduced dataset...")
    btc_dataset = get_dataset("btc-reduced", max_nodes=2000)  # Smaller sample for testing
    data = btc_dataset[0]
    
    print(f"   Dataset: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    print(f"   Features: {data.x.shape}, Labels: {data.y.shape}")
    print(f"   Classes: {data.num_classes} ({', '.join(data.label_names)})")
    
    # Check class distribution
    unique_labels, counts = data.y.unique(return_counts=True)
    print(f"\n   Class distribution:")
    for label_id, count in zip(unique_labels.tolist(), counts.tolist()):
        label_name = data.label_names[label_id]
        print(f"     {label_name}: {count} ({count/len(data.y)*100:.1f}%)")
    
    # Generate embeddings
    print("\n2. Generating VGAE embeddings...")
    try:
        vgae_embedder = get_embedding("vgae", embedding_dim=64, epochs=50, verbose=False)
        embeddings = vgae_embedder.fit_transform(data)
        print(f"   ✓ Embeddings generated: {embeddings.shape}")
    except Exception as e:
        print(f"   ✗ Embedding failed: {e}")
        return False
    
    # Prepare data for classification
    print("\n3. Preparing classification data...")
    X = embeddings  # Already numpy array
    y = data.y.numpy()
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    print("\n4. Training logistic regression classifier...")
    try:
        classifier = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            multi_class="ovr",
            class_weight="balanced",  # Handle class imbalance
            random_state=42
        )
        classifier.fit(X_train_scaled, y_train)
        print("   ✓ Classifier trained")
    except Exception as e:
        print(f"   ✗ Training failed: {e}")
        return False
    
    # Evaluate
    print("\n5. Evaluating classification performance...")
    try:
        y_pred = classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Detailed classification report
        print("\n   Classification Report:")
        report = classification_report(
            y_test, y_pred, 
            target_names=data.label_names,
            zero_division=0
        )
        print(report)
        
    except Exception as e:
        print(f"   ✗ Evaluation failed: {e}")
        return False
    
    return True

def compare_btc_with_cora():
    """Compare BTC classification with Cora to show same workflow."""
    print("\n" + "=" * 60)
    print("Comparing Classification Workflows: BTC vs Cora")
    print("=" * 60)
    
    results = {}
    
    for dataset_name in ["Cora", "btc-reduced"]:
        print(f"\nTesting {dataset_name}...")
        
        try:
            if dataset_name == "Cora":
                dataset = get_dataset("Cora")
                data = dataset.to("torch-geometric")
            else:
                dataset = get_dataset("btc-reduced", max_nodes=1500)
                data = dataset[0]
            
            print(f"  Loaded: {data.num_nodes} nodes, {data.num_classes} classes")
            
            # Quick VGAE embedding
            embedder = get_embedding("vgae", embedding_dim=32, epochs=20, verbose=False)
            embeddings = embedder.fit_transform(data)
            
            # Quick classification test
            X = embeddings  # Already numpy array
            y = data.y.numpy()
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            classifier = LogisticRegression(
                max_iter=1000, 
                solver="lbfgs", 
                class_weight="balanced" if dataset_name == "btc-reduced" else None,
                random_state=42
            )
            classifier.fit(X_train_scaled, y_train)
            
            y_pred = classifier.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[dataset_name] = accuracy
            print(f"  ✓ Classification accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[dataset_name] = None
    
    print(f"\n{'Dataset':<15} {'Accuracy':<10} {'Status'}")
    print("-" * 35)
    for dataset, acc in results.items():
        if acc is not None:
            print(f"{dataset:<15} {acc:.4f}     ✓ Working")
        else:
            print(f"{dataset:<15} {'N/A':<10} ✗ Failed")
    
    return all(acc is not None for acc in results.values())

if __name__ == "__main__":
    print("BTC Dataset Classification Test")
    print("=" * 60)
    
    # Test basic classification
    success = test_btc_classification()
    
    if success:
        # Compare with Cora
        compare_success = compare_btc_with_cora()
        
        if compare_success:
            print("\n" + "=" * 60)
            print("✓ BTC dataset classification pipeline works!")
            print("✓ Same workflow as Cora and other datasets!")
            print("✓ Ready for configuration-based experiments!")
            
            print("\nNext Steps:")
            print("1. Run simple config:")
            print("   python scripts/classification/backup/run_classification.py btc_reduced_simple_config.yaml")
            print()
            print("2. Run full comparison:")
            print("   python scripts/classification/backup/run_classification.py btc_reduced_classification_config.yaml")
            print()
            print("3. Run unified framework:")
            print("   python scripts/classification/backup/run_classification.py btc_reduced_unified_config.yaml")
        else:
            print("\n✗ Comparison test had issues")
    else:
        print("\n✗ Classification test failed")
        sys.exit(1)