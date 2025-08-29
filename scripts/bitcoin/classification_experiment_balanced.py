#!/usr/bin/env python3
"""
Enhanced Classification Experiment with Class Imbalance Handling

Extends the standard classification experiment with multiple sampling strategies
to address severe class imbalance in datasets like BTC-reduced.

Sampling strategies include:
- Random oversampling (ROS)
- SMOTE (Synthetic Minority Over-sampling)
- Random undersampling (RUS)
- ADASYN (Adaptive Synthetic Sampling)
- Combined over/under sampling

Usage:
    python classification_experiment_balanced.py config.yaml
"""

import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add parent directory to path for L2GX imports
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Imbalanced learning libraries
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek

from l2gx.align import get_aligner
from l2gx.datasets import get_dataset
from l2gx.embedding import get_embedding
from l2gx.graphs import TGraph

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class BalancedClassificationExperiment:
    """Classification experiment with advanced class imbalance handling."""

    def __init__(self, config_path: str):
        """Initialize experiment from configuration file."""
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Setup output directory
        self.output_dir = Path(self.config["experiment"]["output_dir"])
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load dataset
        self.graph, self.labels = self._load_dataset()
        
        # Get class distribution info
        self.class_distribution = self._analyze_class_distribution()

        # Results storage
        self.results = []

    def _load_config(self) -> dict:
        """Load and validate configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Add default sampling config if not present
        if "sampling" not in config:
            config["sampling"] = {
                "strategies": ["none", "class_weight", "smote", "ros", "rus"],
                "smote_neighbors": 5,
                "random_state": 42
            }

        return config

    def _load_dataset(self) -> tuple[TGraph, np.ndarray]:
        """Load dataset according to configuration."""
        dataset_config = self.config["dataset"]
        dataset_name = dataset_config["name"]

        print(f"Loading {dataset_name} dataset...")

        # Get dataset with optional parameters
        kwargs = {}
        if "data_root" in dataset_config:
            kwargs["root"] = dataset_config["data_root"]
        if "max_nodes" in dataset_config:
            kwargs["max_nodes"] = dataset_config["max_nodes"]

        dataset = get_dataset(dataset_name, **kwargs)
        
        # Handle different dataset types
        if dataset_name.lower() in ["btc", "btc-reduced", "btc_reduced"]:
            pg_data = dataset[0]
        else:
            pg_data = dataset.to("torch-geometric")

        graph = TGraph(
            edge_index=pg_data.edge_index,
            x=pg_data.x,
            y=pg_data.y,
            num_nodes=pg_data.num_nodes,
        )

        labels = pg_data.y.cpu().numpy()
        
        # Store label names if available
        self.label_names = getattr(pg_data, 'label_names', None)

        print(f"{dataset_name}: {graph.num_nodes} nodes, {graph.num_edges} edges, {len(np.unique(labels))} classes")
        return graph, labels

    def _analyze_class_distribution(self) -> dict:
        """Analyze class distribution in the dataset."""
        unique, counts = np.unique(self.labels, return_counts=True)
        distribution = {}
        
        print("\nClass Distribution:")
        for label, count in zip(unique, counts):
            pct = count / len(self.labels) * 100
            if self.label_names:
                name = self.label_names[label]
                distribution[label] = {"name": name, "count": count, "percentage": pct}
                print(f"  {name:15s}: {count:6d} ({pct:5.1f}%)")
            else:
                distribution[label] = {"name": f"Class_{label}", "count": count, "percentage": pct}
                print(f"  Class {label:2d}: {count:6d} ({pct:5.1f}%)")
        
        # Identify imbalance severity
        max_pct = max(d["percentage"] for d in distribution.values())
        min_pct = min(d["percentage"] for d in distribution.values())
        imbalance_ratio = max_pct / min_pct
        
        print(f"\nImbalance Ratio: {imbalance_ratio:.1f}:1")
        if imbalance_ratio > 10:
            print("  ⚠️ SEVERE class imbalance detected!")
        elif imbalance_ratio > 3:
            print("  ⚠️ Moderate class imbalance detected")
        
        return distribution

    def get_sampler(self, strategy: str):
        """Get the appropriate sampler for the given strategy."""
        sampling_config = self.config["sampling"]
        random_state = sampling_config.get("random_state", 42)
        
        if strategy == "none" or strategy == "class_weight":
            return None
            
        elif strategy == "ros":
            # Random Over-Sampling
            return RandomOverSampler(random_state=random_state)
            
        elif strategy == "smote":
            # SMOTE with adjustable neighbors for small classes
            k_neighbors = min(sampling_config.get("smote_neighbors", 5), 
                            min(self.class_distribution[k]["count"] for k in self.class_distribution) - 1)
            k_neighbors = max(1, k_neighbors)
            return SMOTE(k_neighbors=k_neighbors, random_state=random_state)
            
        elif strategy == "borderline_smote":
            # Borderline SMOTE - focuses on boundary examples
            k_neighbors = min(5, min(self.class_distribution[k]["count"] for k in self.class_distribution) - 1)
            k_neighbors = max(1, k_neighbors)
            return BorderlineSMOTE(k_neighbors=k_neighbors, random_state=random_state)
            
        elif strategy == "adasyn":
            # Adaptive Synthetic Sampling
            n_neighbors = min(5, min(self.class_distribution[k]["count"] for k in self.class_distribution) - 1)
            n_neighbors = max(1, n_neighbors)
            return ADASYN(n_neighbors=n_neighbors, random_state=random_state)
            
        elif strategy == "rus":
            # Random Under-Sampling
            return RandomUnderSampler(random_state=random_state)
            
        elif strategy == "tomek":
            # Remove Tomek links (cleans boundaries)
            return TomekLinks()
            
        elif strategy == "enn":
            # Edited Nearest Neighbors (removes noisy samples)
            return EditedNearestNeighbours()
            
        elif strategy == "smote_tomek":
            # Combined: SMOTE then remove Tomek links
            return SMOTETomek(random_state=random_state)
            
        elif strategy == "smote_enn":
            # Combined: SMOTE then Edited Nearest Neighbors
            return SMOTEENN(random_state=random_state)
            
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    def compute_embedding(self, method: str, embedding_dim: int) -> tuple[np.ndarray, float]:
        """Generate embedding based on method configuration."""
        start_time = time.time()
        method_config = self.config["methods"][method]

        # Same as original classification_experiment.py
        if "hierarchical" in method:
            embedding = self._compute_hierarchical_embedding(embedding_dim, method_config)
        elif method == "full_graph":
            embedding = self._compute_full_graph_embedding(embedding_dim, method_config)
        else:
            embedding = self._compute_patched_embedding(embedding_dim, method_config)

        return embedding, time.time() - start_time

    def _compute_full_graph_embedding(self, embedding_dim: int, method_config: dict) -> np.ndarray:
        """Compute full graph embedding."""
        embed_config = method_config["base_embedding"]

        embedder = get_embedding(
            embed_config["method"],
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim * 2,
            epochs=embed_config["epochs"],
            learning_rate=embed_config["learning_rate"],
            patience=embed_config["patience"],
            verbose=embed_config.get("verbose", False)
        )

        return embedder.fit_transform(self.graph.to_tg())

    def _compute_hierarchical_embedding(self, embedding_dim: int, method_config: dict) -> np.ndarray:
        """Compute hierarchical embedding."""
        hier_config = method_config["hierarchical"]
        embed_config = method_config["base_embedding"]

        aligner = get_aligner("l2g")

        embedder = get_embedding(
            "hierarchical",
            embedding_dim=embedding_dim,
            aligner=aligner,
            max_patch_size=hier_config["max_patch_size"],
            base_method=embed_config["method"],
            min_overlap=hier_config.get("min_overlap", 64),
            target_overlap=hier_config.get("target_overlap", 128),
            epochs=embed_config.get("epochs", 100),
            learning_rate=embed_config.get("learning_rate", 0.001),
            patience=embed_config.get("patience", 20),
            verbose=False
        )

        return embedder.fit_transform(self.graph.to_tg())

    def _compute_patched_embedding(self, embedding_dim: int, method_config: dict) -> np.ndarray:
        """Compute patched embedding with alignment."""
        patch_config = method_config["patches"]
        align_config = method_config["alignment"]
        embed_config = method_config["base_embedding"]

        # Create aligner based on configuration
        if align_config["method"] == "l2g":
            aligner = get_aligner("l2g")
            if "randomized_method" in align_config:
                aligner.randomized_method = align_config["randomized_method"]
            if "sketch_method" in align_config:
                aligner.sketch_method = align_config["sketch_method"]

        elif align_config["method"] == "geo":
            geo_kwargs = {
                "method": align_config.get("geo_method", "orthogonal"),
                "use_scale": align_config.get("use_scale", True),
                "verbose": align_config.get("verbose", False)
            }

            if align_config.get("use_randomized_init", False):
                geo_kwargs["use_randomized_init"] = True
                geo_kwargs["randomized_method"] = align_config.get("randomized_method", "randomized")

            aligner = get_aligner("geo", **geo_kwargs)
        else:
            raise ValueError(f"Unknown alignment method: {align_config['method']}")

        embedder = get_embedding(
            "patched",
            embedding_dim=embedding_dim,
            aligner=aligner,
            num_patches=patch_config["num_patches"],
            base_method=embed_config["method"],
            clustering_method=patch_config.get("clustering_method", "metis"),
            min_overlap=patch_config.get("min_overlap", 256),
            target_overlap=patch_config.get("target_overlap", 512),
            sparsify_method=patch_config.get("sparsify_method", "resistance"),
            target_patch_degree=patch_config.get("target_patch_degree", 4),
            epochs=embed_config["epochs"],
            learning_rate=embed_config["learning_rate"],
            patience=embed_config["patience"],
            verbose=embed_config.get("verbose", False)
        )

        return embedder.fit_transform(self.graph.to_tg())

    def run_classification(self, embedding: np.ndarray, sampling_strategy: str) -> dict:
        """Run classification with specified sampling strategy."""
        params = self.config["parameters"]
        class_config = self.config["classification"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embedding, self.labels,
            test_size=params["test_size"],
            random_state=params["random_seed"],
            stratify=self.labels
        )

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Apply sampling strategy
        sampler = self.get_sampler(sampling_strategy)
        if sampler is not None:
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
            print(f"    Resampled training set: {len(y_train)} → {len(y_train_resampled)} samples")
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        # Create classifier
        classifier_type = class_config["classifier"]
        classifier_params = class_config["params"].copy()
        
        # Handle class weights for logistic regression
        if sampling_strategy == "class_weight" and classifier_type == "logistic_regression":
            classifier_params["class_weight"] = "balanced"
        
        if classifier_type == "logistic_regression":
            classifier = LogisticRegression(**classifier_params)
        elif classifier_type == "random_forest":
            classifier = RandomForestClassifier(**classifier_params)
        elif classifier_type == "svm":
            classifier = SVC(**classifier_params)
        else:
            raise ValueError(f"Unknown classifier: {classifier_type}")

        # Train and evaluate
        classifier.fit(X_train_resampled, y_train_resampled)
        y_pred = classifier.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0
        )
        
        # Weighted and macro F1
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "per_class_precision": precision,
            "per_class_recall": recall,
            "per_class_f1": f1,
            "predictions": y_pred,
            "true_labels": y_test
        }

    def run_single_experiment(self, method: str, embedding_dim: int, 
                            sampling_strategy: str, run_id: int) -> dict:
        """Run single experiment for given configuration."""
        print(f"    Run {run_id+1}: {method}, dim={embedding_dim}, sampling={sampling_strategy}")

        # Generate embedding
        try:
            embedding, embed_time = self.compute_embedding(method, embedding_dim)
        except Exception as e:
            print(f"      Warning: Embedding failed ({e})")
            return None

        # Run classification with sampling
        classification_start = time.time()
        metrics = self.run_classification(embedding, sampling_strategy)
        classification_time = time.time() - classification_start

        return {
            "method": method,
            "embedding_dim": embedding_dim,
            "sampling_strategy": sampling_strategy,
            "run_id": run_id,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "f1_weighted": metrics["f1_weighted"],
            "f1_macro": metrics["f1_macro"],
            "embedding_time": embed_time,
            "classification_time": classification_time,
        }

    def run(self):
        """Run the complete experiment."""
        print(f"Starting {self.config['experiment']['name']}:")
        print(f"  Dataset: {self.config['dataset']['name']}")
        
        methods = [m for m, cfg in self.config["methods"].items() if cfg.get("enabled", True)]
        dimensions = self.config["parameters"]["dimensions"]
        sampling_strategies = self.config["sampling"]["strategies"]
        n_runs = self.config["parameters"]["n_runs"]
        
        print(f"  Methods: {len(methods)} ({', '.join(methods)})")
        print(f"  Dimensions: {len(dimensions)} ({dimensions})")
        print(f"  Sampling strategies: {len(sampling_strategies)} ({', '.join(sampling_strategies)})")
        print(f"  Runs per configuration: {n_runs}")
        
        total_experiments = len(methods) * len(dimensions) * len(sampling_strategies) * n_runs
        print(f"  Total experiments: {total_experiments}")
        print("=" * 80)

        completed = 0
        for method in methods:
            print(f"\nRunning {method} experiments...")
            print(f"  Description: {self.config['methods'][method].get('description', '')}")

            for dim in dimensions:
                print(f"\n  Dimension {dim}:")
                for strategy in sampling_strategies:
                    accuracies = []
                    balanced_accs = []
                    f1_macros = []
                    
                    for run_id in range(n_runs):
                        result = self.run_single_experiment(method, dim, strategy, run_id)
                        
                        if result:
                            self.results.append(result)
                            accuracies.append(result["accuracy"])
                            balanced_accs.append(result["balanced_accuracy"])
                            f1_macros.append(result["f1_macro"])
                        
                        completed += 1
                        progress = completed / total_experiments * 100
                    
                    if accuracies:
                        print(f"    {strategy:15s}: {np.mean(accuracies):.4f} acc, "
                              f"{np.mean(balanced_accs):.4f} bal_acc, "
                              f"{np.mean(f1_macros):.4f} f1_macro "
                              f"({progress:.1f}% complete)")

        self.save_results()
        self.create_summary()
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETE")

    def save_results(self):
        """Save experiment results."""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(self.output_dir / "raw_results.csv", index=False)
            
            # Create summary by method, dimension, and sampling strategy
            summary_df = df.groupby(["method", "embedding_dim", "sampling_strategy"]).agg({
                "accuracy": ["mean", "std"],
                "balanced_accuracy": ["mean", "std"],
                "f1_weighted": ["mean", "std"],
                "f1_macro": ["mean", "std"],
                "embedding_time": "mean",
                "classification_time": "mean"
            }).round(4)
            
            summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
            summary_df.to_csv(self.output_dir / "summary_results.csv")
            print(f"\nResults saved to {self.output_dir}/")

    def create_summary(self):
        """Create experiment summary report."""
        df = pd.DataFrame(self.results)
        
        # Find best configuration
        best_idx = df.groupby(["method", "embedding_dim", "sampling_strategy"])["balanced_accuracy"].mean().idxmax()
        best_config = df[
            (df["method"] == best_idx[0]) & 
            (df["embedding_dim"] == best_idx[1]) & 
            (df["sampling_strategy"] == best_idx[2])
        ]
        
        report = f"""
EXPERIMENT SUMMARY
{'='*60}
Dataset: {self.config['dataset']['name']}
Methods tested: {len(df['method'].unique())}
Dimensions tested: {sorted(df['embedding_dim'].unique())}
Sampling strategies: {sorted(df['sampling_strategy'].unique())}
Total runs: {len(df)}

BEST CONFIGURATION
{'='*60}
Method: {best_idx[0]}
Dimension: {best_idx[1]}
Sampling: {best_idx[2]}
Balanced Accuracy: {best_config['balanced_accuracy'].mean():.4f} ± {best_config['balanced_accuracy'].std():.4f}
F1 Macro: {best_config['f1_macro'].mean():.4f} ± {best_config['f1_macro'].std():.4f}

TOP CONFIGURATIONS BY BALANCED ACCURACY
{'='*60}
"""
        # Get top 10 configurations
        top_configs = df.groupby(["method", "embedding_dim", "sampling_strategy"]).agg({
            "balanced_accuracy": "mean",
            "f1_macro": "mean"
        }).sort_values("balanced_accuracy", ascending=False).head(10)
        
        for idx, row in top_configs.iterrows():
            report += f"{idx[0]:20s} dim={idx[1]:3d} {idx[2]:15s}: {row['balanced_accuracy']:.4f} bal_acc, {row['f1_macro']:.4f} f1\n"
        
        # Save report
        with open(self.output_dir / "experiment_summary.txt", "w") as f:
            f.write(report)
        
        print(report)


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python classification_experiment_balanced.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    experiment = BalancedClassificationExperiment(config_path)
    experiment.run()


if __name__ == "__main__":
    main()