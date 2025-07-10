#!/usr/bin/env python3
"""
Node Classification Experiment

This script uses the EmbeddingExperiment class to generate embeddings and then
performs node classification using logistic regression with proper train/test splits.

Usage: python run/node_classification.py <config_file>

Example:
    python run/node_classification.py configs/cora_patched.yaml
"""

import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import core embedding functionality
from embedding_experiment import EmbeddingExperiment

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class NodeClassificationExperiment:
    """Handles node classification experiments using embeddings"""
    
    def __init__(self, config_path: str):
        """Initialize with embedding experiment config"""
        self.config_path = Path(config_path)
        self.embedding_experiment = EmbeddingExperiment(config_path)
        
        # Classification parameters
        self.test_size = 0.2  # 80/20 train/test split
        self.random_state = 42  # For reproducible results
        self.max_iter = 1000  # For logistic regression convergence
        
        # Results storage
        self.results = {}
        
    def create_train_test_split(self, embedding: np.ndarray, labels: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create train/test split for node classification
        
        Args:
            embedding: Node embeddings (N x d)
            labels: Node labels (N,)
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"\\n🔀 Creating train/test split...")
        
        # Convert labels to numpy if needed
        if hasattr(labels, 'cpu'):
            y = labels.cpu().numpy()
        else:
            y = labels
        
        # Filter out unlabeled nodes (if any)
        labeled_mask = y >= 0
        X_labeled = embedding[labeled_mask]
        y_labeled = y[labeled_mask]
        
        print(f"Dataset: {len(X_labeled)} labeled nodes, {len(np.unique(y_labeled))} classes")
        
        # Create stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_labeled, y_labeled,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_labeled  # Ensure balanced class distribution
        )
        
        print(f"Split: {len(X_train)} train, {len(X_test)} test")
        print(f"Train classes: {np.bincount(y_train)}")
        print(f"Test classes: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_classifier(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """
        Train logistic regression classifier
        
        Args:
            X_train: Training embeddings
            y_train: Training labels
            
        Returns:
            Trained classifier
        """
        print(f"\\n🧠 Training logistic regression classifier...")
        
        # Standardize features (important for logistic regression)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train logistic regression with L2 regularization
        classifier = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver='lbfgs',  # Good for small datasets
            multi_class='ovr'  # One-vs-rest for multi-class
        )
        
        classifier.fit(X_train_scaled, y_train)
        
        # Training accuracy
        train_pred = classifier.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        return classifier
    
    def evaluate_classifier(self, classifier: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate classifier on test set
        
        Args:
            classifier: Trained classifier
            X_test: Test embeddings
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\\n📊 Evaluating classifier...")
        
        # Scale test features using training scaler
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = classifier.predict(X_test_scaled)
        y_pred_proba = classifier.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Per-class metrics
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'num_test_samples': len(y_test),
            'num_classes': len(np.unique(y_test))
        }
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"F1 Score (macro): {f1_macro:.4f}")
        print(f"F1 Score (micro): {f1_micro:.4f}")
        print(f"F1 Score (weighted): {f1_weighted:.4f}")
        
        return results
    
    def print_detailed_results(self, eval_results: Dict[str, Any]) -> None:
        """Print detailed classification results"""
        print(f"\\n📈 Detailed Classification Results:")
        print("=" * 50)
        
        # Overall metrics
        print(f"Test Samples: {eval_results['num_test_samples']}")
        print(f"Number of Classes: {eval_results['num_classes']}")
        print(f"Overall Accuracy: {eval_results['accuracy']:.4f}")
        print(f"F1 Score (macro): {eval_results['f1_macro']:.4f}")
        print(f"F1 Score (weighted): {eval_results['f1_weighted']:.4f}")
        print()
        
        # Per-class results
        print("Per-Class Results:")
        print("-" * 30)
        class_report = eval_results['classification_report']
        
        for class_id in sorted([k for k in class_report.keys() if k.isdigit()], key=int):
            metrics = class_report[class_id]
            print(f"Class {class_id}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}, "
                  f"Support={metrics['support']}")
        
        print()
        print("Confusion Matrix:")
        print("-" * 20)
        conf_matrix = np.array(eval_results['confusion_matrix'])
        print(conf_matrix)
    
    def save_results(self, eval_results: Dict[str, Any], embedding_results: Dict[str, Any]) -> None:
        """Save classification results"""
        print(f"\\n💾 Saving classification results...")
        
        output_dir = self.embedding_experiment.output_dir
        
        # Combine all results
        combined_results = {
            'classification': eval_results,
            'embedding': embedding_results,
            'experimental_setup': {
                'test_size': self.test_size,
                'random_state': self.random_state,
                'max_iter': self.max_iter,
                'config_file': str(self.config_path)
            }
        }
        
        # Save detailed results as YAML
        import yaml
        with open(output_dir / "classification_results.yaml", 'w') as f:
            yaml.dump(combined_results, f, default_flow_style=False)
        
        # Save human-readable summary
        with open(output_dir / "classification_summary.txt", 'w') as f:
            f.write(f"Node Classification Results\\n")
            f.write("=" * 40 + "\\n\\n")
            
            f.write(f"Dataset: {embedding_results['dataset']['name']}\\n")
            f.write(f"Embedding Type: {embedding_results['embedding_type']}\\n")
            f.write(f"Embedding Dimension: {embedding_results['embedding']['shape'][1]}\\n")
            if embedding_results['embedding_type'] == 'patched_l2g':
                f.write(f"Number of Patches: {embedding_results['num_patches']}\\n")
            f.write(f"\\n")
            
            f.write(f"Classification Setup:\\n")
            f.write(f"  Train/Test Split: {int((1-self.test_size)*100)}/{int(self.test_size*100)}\\n")
            f.write(f"  Classifier: Logistic Regression\\n")
            f.write(f"  Feature Scaling: StandardScaler\\n")
            f.write(f"\\n")
            
            f.write(f"Results:\\n")
            f.write(f"  Test Accuracy: {eval_results['accuracy']:.4f}\\n")
            f.write(f"  F1 Score (macro): {eval_results['f1_macro']:.4f}\\n")
            f.write(f"  F1 Score (weighted): {eval_results['f1_weighted']:.4f}\\n")
            f.write(f"  Number of Test Samples: {eval_results['num_test_samples']}\\n")
            f.write(f"  Number of Classes: {eval_results['num_classes']}\\n")
        
        print(f"✅ Classification results saved to {output_dir}/")
    
    def run_experiment(self) -> Dict[str, Any]:
        """
        Run complete node classification experiment
        
        Returns:
            Dictionary of results
        """
        print(f"🔬 STARTING NODE CLASSIFICATION EXPERIMENT")
        print("=" * 80)
        print(f"Config: {self.config_path}")
        print(f"Output: {self.embedding_experiment.output_dir}")
        
        # Step 1: Generate embeddings
        print(f"\\n1️⃣ Generating embeddings...")
        embedding, patches, data = self.embedding_experiment.run_experiment()
        
        # Step 2: Create train/test split
        X_train, X_test, y_train, y_test = self.create_train_test_split(embedding, data.y)
        
        # Step 3: Train classifier
        classifier = self.train_classifier(X_train, y_train)
        
        # Step 4: Evaluate classifier
        eval_results = self.evaluate_classifier(classifier, X_test, y_test)
        
        # Step 5: Print detailed results
        self.print_detailed_results(eval_results)
        
        # Step 6: Save results
        self.save_results(eval_results, self.embedding_experiment.results)
        
        print(f"\\n✅ NODE CLASSIFICATION EXPERIMENT COMPLETE!")
        print(f"Final Accuracy: {eval_results['accuracy']:.4f}")
        
        return eval_results


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python run/node_classification.py <config_file>")
        print("Example: python run/node_classification.py configs/cora_patched.yaml")
        print()
        print("Available configs:")
        print("  configs/cora_patched.yaml  - Cora with L2G patches")
        print("  configs/cora_whole.yaml    - Cora whole graph")
        print("  configs/pubmed_patched.yaml - PubMed with L2G patches")
        print("  configs/dgi_patched.yaml   - Cora with DGI method")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    try:
        # Run node classification experiment
        experiment = NodeClassificationExperiment(config_file)
        results = experiment.run_experiment()
        
        print(f"\\n🎯 FINAL RESULTS:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 (macro): {results['f1_macro']:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()