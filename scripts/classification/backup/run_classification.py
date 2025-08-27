#!/usr/bin/env python3
"""
Node Classification Pipeline

Performs node classification using embeddings generated from config files.
Supports multiple classifiers and evaluation strategies.

Usage:
    python run_classification.py <embedding_config> [--classifier <config>]

Example:
    python run_classification.py ../embedding/embedding_config_l2g.yaml
    python run_classification.py ../embedding/embedding_config_geo.yaml --classifier classification_config.yaml
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import the embedding experiment runner
sys.path.insert(0, str(Path(__file__).parent.parent / "embedding"))
from run_embedding_config import ConfigurableEmbeddingExperiment

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class NodeClassificationPipeline:
    """Pipeline for node classification using pre-computed or generated embeddings."""

    def __init__(self, embedding_config: str, classification_config: str | None = None):
        """
        Initialize classification pipeline.
        
        Args:
            embedding_config: Path to embedding configuration YAML
            classification_config: Optional path to classification configuration YAML
        """
        self.embedding_config_path = Path(embedding_config)
        self.classification_config_path = Path(classification_config) if classification_config else None

        # Load configurations
        self.embedding_config = self._load_yaml(self.embedding_config_path)
        self.classification_config = self._load_classification_config()

        # Setup output directory
        self.output_dir = Path("results") / f"classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store results
        self.results = {}

    def _load_yaml(self, path: Path) -> dict:
        """Load YAML configuration file."""
        with open(path) as f:
            return yaml.safe_load(f)

    def _load_classification_config(self) -> dict:
        """Load or create default classification configuration."""
        if self.classification_config_path and self.classification_config_path.exists():
            return self._load_yaml(self.classification_config_path)
        else:
            # Default configuration
            return {
                "classifier": {
                    "type": "logistic_regression",
                    "params": {
                        "max_iter": 1000,
                        "random_state": 42,
                        "solver": "lbfgs",
                        "multi_class": "auto"
                    }
                },
                "evaluation": {
                    "test_size": 0.2,
                    "val_size": 0.1,
                    "stratify": True,
                    "random_state": 42,
                    "cross_validation": {
                        "enabled": False,
                        "folds": 5
                    }
                },
                "preprocessing": {
                    "scale_features": True,
                    "handle_imbalanced": False
                }
            }

    def get_embeddings(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Get embeddings either by loading or generating them.

        Returns:
            embeddings, labels, metadata
        """
        print("\n" + "="*60)
        print("OBTAINING EMBEDDINGS")
        print("="*60)

        # Check if embeddings already exist
        embedding_dir = Path(self.embedding_config["experiment"]["output_dir"])
        npz_file = embedding_dir / "embedding_results.npz"

        if npz_file.exists():
            print(f"Loading existing embeddings from {npz_file}")
            data = np.load(npz_file, allow_pickle=True)
            embeddings = data["embedding"]
            labels = data["labels"]

            # Load metadata safely
            metadata_file = embedding_dir / "experiment_metadata.yaml"
            if metadata_file.exists():
                try:
                    metadata = self._load_yaml(metadata_file)
                except yaml.constructor.ConstructorError:
                    # Some metadata may have Python-specific types
                    metadata = {"loaded_from": str(npz_file), "note": "Metadata had incompatible types"}
            else:
                metadata = {"loaded_from": str(npz_file)}

        else:
            print("Generating new embeddings...")
            # Run embedding experiment
            experiment = ConfigurableEmbeddingExperiment(str(self.embedding_config_path))
            embedding, _, data = experiment.run_experiment()

            embeddings = embedding
            labels = data.y.cpu().numpy()
            metadata = experiment.results

        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Number of classes: {len(np.unique(labels))}")

        return embeddings, labels, metadata

    def create_classifier(self) -> object:
        """Create classifier based on configuration."""
        classifier_config = self.classification_config["classifier"]
        classifier_type = classifier_config["type"]
        params = classifier_config.get("params", {})

        if classifier_type == "logistic_regression":
            return LogisticRegression(**params)
        elif classifier_type == "random_forest":
            return RandomForestClassifier(**params)
        elif classifier_type == "svm":
            return SVC(**params)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def preprocess_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Preprocess features based on configuration."""
        if self.classification_config["preprocessing"]["scale_features"]:
            if fit:
                self.scaler = StandardScaler()
                return self.scaler.fit_transform(X)
            else:
                return self.scaler.transform(X)
        return X

    def split_data(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Split data into train/val/test sets.

        Returns:
            Dictionary with split data
        """
        eval_config = self.classification_config["evaluation"]

        # Filter out unlabeled nodes if any
        mask = y >= 0
        X = X[mask]
        y = y[mask]

        # First split: train+val vs test
        test_size = eval_config["test_size"]
        stratify = y if eval_config["stratify"] else None

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=eval_config["random_state"],
            stratify=stratify
        )

        # Second split: train vs val
        val_size = eval_config["val_size"] / (1 - test_size)  # Adjust for remaining data
        stratify_val = y_trainval if eval_config["stratify"] else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_size,
            random_state=eval_config["random_state"],
            stratify=stratify_val
        )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test
        }

    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray, split_name: str) -> dict:
        """Evaluate model on given data split."""
        y_pred = model.predict(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "f1_macro": f1_score(y, y_pred, average="macro"),
            "f1_micro": f1_score(y, y_pred, average="micro"),
            "f1_weighted": f1_score(y, y_pred, average="weighted"),
            "classification_report": classification_report(y, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist()
        }

        print(f"\n{split_name} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")

        return metrics

    def run_classification(self):
        """Run the complete classification pipeline."""
        print("\n" + "="*60)
        print("NODE CLASSIFICATION PIPELINE")
        print("="*60)

        # Step 1: Get embeddings
        X, y, embedding_metadata = self.get_embeddings()

        # Step 2: Split data
        print("\nSplitting data...")
        splits = self.split_data(X, y)
        print(f"Train: {len(splits['y_train'])}, Val: {len(splits['y_val'])}, Test: {len(splits['y_test'])}")

        # Step 3: Preprocess features
        print("\nPreprocessing features...")
        splits["X_train"] = self.preprocess_features(splits["X_train"], fit=True)
        splits["X_val"] = self.preprocess_features(splits["X_val"], fit=False)
        splits["X_test"] = self.preprocess_features(splits["X_test"], fit=False)

        # Step 4: Train classifier
        print(f"\nTraining {self.classification_config['classifier']['type']} classifier...")
        classifier = self.create_classifier()
        classifier.fit(splits["X_train"], splits["y_train"])

        # Step 5: Evaluate on all splits
        results = {
            "train": self.evaluate_model(classifier, splits["X_train"], splits["y_train"], "Train"),
            "val": self.evaluate_model(classifier, splits["X_val"], splits["y_val"], "Validation"),
            "test": self.evaluate_model(classifier, splits["X_test"], splits["y_test"], "Test")
        }

        # Step 6: Cross-validation if enabled
        cv_config = self.classification_config["evaluation"]["cross_validation"]
        if cv_config["enabled"]:
            print(f"\nRunning {cv_config['folds']}-fold cross-validation...")
            X_preprocessed = self.preprocess_features(X, fit=True)
            cv_scores = cross_val_score(
                self.create_classifier(), X_preprocessed, y,
                cv=cv_config["folds"],
                scoring="accuracy"
            )
            results["cross_validation"] = {
                "scores": cv_scores.tolist(),
                "mean": cv_scores.mean(),
                "std": cv_scores.std()
            }
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Store all results
        self.results = {
            "classification_results": results,
            "embedding_metadata": embedding_metadata,
            "configuration": {
                "embedding_config": self.embedding_config,
                "classification_config": self.classification_config
            },
            "timestamp": datetime.now().isoformat()
        }

        # Save results
        self.save_results()

        return results

    def save_results(self):
        """Save all results to files."""
        print(f"\nSaving results to {self.output_dir}")

        # Save full results as YAML
        with open(self.output_dir / "results.yaml", "w") as f:
            yaml.dump(self.results, f, default_flow_style=False)

        # Save summary
        with open(self.output_dir / "summary.txt", "w") as f:
            f.write("NODE CLASSIFICATION RESULTS\n")
            f.write("="*50 + "\n\n")

            # Dataset info
            if "dataset" in self.results["embedding_metadata"]:
                dataset_info = self.results["embedding_metadata"]["dataset"]
                f.write(f"Dataset: {dataset_info['name']}\n")
                f.write(f"Nodes: {dataset_info['num_nodes']}\n")
                f.write(f"Classes: {dataset_info['num_classes']}\n\n")

            # Embedding info
            f.write("Embedding Configuration:\n")
            embed_config = self.embedding_config["embedding"]
            f.write(f"  Method: {embed_config['method']}\n")
            f.write(f"  Dimension: {embed_config['embedding_dim']}\n")

            align_config = self.embedding_config["alignment"]
            f.write(f"  Alignment: {align_config['method']}\n\n")

            # Classification info
            f.write("Classification Configuration:\n")
            class_config = self.classification_config["classifier"]
            f.write(f"  Classifier: {class_config['type']}\n")
            f.write(f"  Feature scaling: {self.classification_config['preprocessing']['scale_features']}\n\n")

            # Results
            f.write("Results:\n")
            test_results = self.results["classification_results"]["test"]
            f.write(f"  Test Accuracy: {test_results['accuracy']:.4f}\n")
            f.write(f"  Test F1 (macro): {test_results['f1_macro']:.4f}\n")
            f.write(f"  Test F1 (weighted): {test_results['f1_weighted']:.4f}\n")

            if "cross_validation" in self.results["classification_results"]:
                cv = self.results["classification_results"]["cross_validation"]
                f.write(f"  CV Accuracy: {cv['mean']:.4f} (+/- {cv['std']:.4f})\n")

        print(f"Results saved to {self.output_dir}/")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run node classification using embeddings from config"
    )
    parser.add_argument(
        "embedding_config",
        help="Path to embedding configuration YAML file"
    )
    parser.add_argument(
        "--classifier",
        help="Path to classification configuration YAML file (optional)"
    )

    args = parser.parse_args()

    try:
        # Run classification pipeline
        pipeline = NodeClassificationPipeline(
            args.embedding_config,
            args.classifier
        )
        results = pipeline.run_classification()

        # Print final summary
        print("\n" + "="*60)
        print("CLASSIFICATION COMPLETE")
        print("="*60)
        test_results = results["test"]
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"Test F1 (macro): {test_results['f1_macro']:.4f}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
