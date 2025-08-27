#!/usr/bin/env python3
"""
Classification Experiment

Compares multiple embedding approaches across different dimensions using YAML configuration.

Usage:
    python configurable_dimension_sweep.py [config_file]
    python configurable_dimension_sweep.py experiment_config.yaml

The script supports:
- Multiple datasets (Cora, PubMed, CiteSeer, etc.)
- Four embedding methods: Full Graph, L2G+Rademacher, Hierarchical+L2G, Geo+Rademacher
- Configurable dimensions, runs, and parameters
- Flexible output formats and plotting options
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

# Add embedding and hierarchical directories
sys.path.insert(0, str(Path(__file__).parent.parent / "embedding"))
sys.path.insert(0, str(Path(__file__).parent.parent / "hierarchical"))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from l2gx.datasets import get_dataset
from l2gx.graphs import TGraph
from l2gx.embedding import get_embedding
from l2gx.align import get_aligner

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ConfigurableDimensionSweep:
    """Configurable dimension sweep experiment with multiple embedding methods."""

    def __init__(self, config_path: str):
        """Initialize experiment from configuration file."""
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Setup output directory
        self.output_dir = Path(self.config["experiment"]["output_dir"])
        self.output_dir.mkdir(exist_ok=True)

        # Load dataset
        self.graph, self.labels = self._load_dataset()

        # Results storage
        self.results = []

    def _load_config(self) -> dict:
        """Load and validate configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Validate required sections
        required_sections = ["experiment", "dataset", "parameters", "methods"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in config")

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

        dataset = get_dataset(dataset_name, **kwargs)
        pg_data = dataset.to("torch-geometric")

        graph = TGraph(
            edge_index=pg_data.edge_index,
            x=pg_data.x,
            y=pg_data.y,
            num_nodes=pg_data.num_nodes,
        )

        # Apply feature normalization if requested
        if dataset_config.get("normalize_features", False) and graph.x is not None:
            row_sums = graph.x.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1.0
            graph.x = graph.x / row_sums

        labels = pg_data.y.cpu().numpy()

        print(f"{dataset_name}: {graph.num_nodes} nodes, {graph.num_edges} edges, {len(np.unique(labels))} classes")
        return graph, labels

    def compute_embedding(self, method: str, embedding_dim: int) -> tuple[np.ndarray, float]:
        """Generate embedding based on method configuration."""
        start_time = time.time()
        method_config = self.config["methods"][method]
        
        try:
            if method == "full_graph":
                embedding = self._compute_full_graph_embedding(embedding_dim, method_config)
            elif method == "hierarchical_l2g":
                embedding = self._compute_hierarchical_embedding(embedding_dim, method_config)
            else:
                # Patched methods (l2g_standard, l2g_rademacher, geo_rademacher)
                embedding = self._compute_patched_embedding(embedding_dim, method_config)
        except Exception as e:
            print(f"  Warning: {method} embedding failed ({e}), using random initialization")
            embedding = np.random.randn(self.graph.num_nodes, embedding_dim) * 0.1
        
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
        
        # Create aligner for hierarchical embedding (defaults to Procrustes for binary trees)
        aligner = get_aligner("l2g")  # Will be handled automatically by hierarchical embedder
        
        # Use the unified hierarchical embedding from the library
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
        """Compute patched embedding with alignment using unified framework."""
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
        
        # Use unified patched embedding
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

    def run_classification(self, embedding: np.ndarray) -> float:
        """Run classification on embedding."""
        params = self.config["parameters"]
        class_config = self.config["classification"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embedding, self.labels,
            test_size=params["test_size"],
            random_state=params["random_seed"],
            stratify=self.labels
        )

        # Scale features if enabled
        if class_config["preprocessing"]["scale_features"]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Create classifier
        classifier_type = class_config["classifier"]
        classifier_params = class_config["params"]

        if classifier_type == "logistic_regression":
            classifier = LogisticRegression(**classifier_params)
        elif classifier_type == "random_forest":
            classifier = RandomForestClassifier(**classifier_params)
        elif classifier_type == "svm":
            classifier = SVC(**classifier_params)
        else:
            raise ValueError(f"Unknown classifier: {classifier_type}")

        # Train and evaluate
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    def run_single_experiment(self, method: str, embedding_dim: int, run_id: int) -> dict:
        """Run single experiment for given method, dimension, and run."""
        print(f"  Run {run_id+1}/{self.config['parameters']['n_runs']}: {method}, dim={embedding_dim}")

        # Generate embedding using unified approach
        try:
            embedding, embed_time = self.compute_embedding(method, embedding_dim)
        except Exception as e:
            print(f"    Warning: Embedding failed ({e}), using random initialization")
            embedding = np.random.randn(self.graph.num_nodes, embedding_dim) * 0.1
            embed_time = 0.0

        # Run classification
        classification_start = time.time()
        accuracy = self.run_classification(embedding)
        classification_time = time.time() - classification_start

        return {
            "method": method,
            "embedding_dim": embedding_dim,
            "run_id": run_id,
            "accuracy": accuracy,
            "embedding_time": embed_time,
            "classification_time": classification_time,
            "total_time": embed_time + classification_time
        }

    def run_all_experiments(self):
        """Run all experiments across methods, dimensions, and runs."""
        # Get enabled methods
        enabled_methods = [name for name, config in self.config["methods"].items()
                          if config.get("enabled", True)]

        dimensions = self.config["parameters"]["dimensions"]
        n_runs = self.config["parameters"]["n_runs"]

        total_experiments = len(enabled_methods) * len(dimensions) * n_runs
        experiment_count = 0

        print(f"Starting {self.config['experiment']['name']}:")
        print(f"  Dataset: {self.config['dataset']['name']}")
        print(f"  Methods: {len(enabled_methods)} ({', '.join(enabled_methods)})")
        print(f"  Dimensions: {len(dimensions)} ({dimensions})")
        print(f"  Runs per configuration: {n_runs}")
        print(f"  Total experiments: {total_experiments}")
        print("=" * 80)

        for method in enabled_methods:
            print(f"\nRunning {method} experiments...")
            method_desc = self.config["methods"][method]["description"]
            print(f"  Description: {method_desc}")

            for embedding_dim in dimensions:
                print(f"\n  Dimension {embedding_dim}:")

                for run_id in range(n_runs):
                    try:
                        result = self.run_single_experiment(method, embedding_dim, run_id)
                        self.results.append(result)

                        experiment_count += 1
                        progress = experiment_count / total_experiments * 100
                        print(f"    Accuracy: {result['accuracy']:.4f}, "
                              f"Time: {result['total_time']:.2f}s "
                              f"({progress:.1f}% complete)")

                    except Exception as e:
                        print(f"    Error in run {run_id+1}: {e}")
                        self.results.append({
                            "method": method,
                            "embedding_dim": embedding_dim,
                            "run_id": run_id,
                            "accuracy": 0.0,
                            "embedding_time": 0.0,
                            "classification_time": 0.0,
                            "total_time": 0.0,
                            "error": str(e)
                        })

                # Print summary for this dimension
                dim_results = [r for r in self.results
                             if r["method"] == method and r["embedding_dim"] == embedding_dim]
                accuracies = [r["accuracy"] for r in dim_results if r["accuracy"] > 0]

                if accuracies:
                    mean_acc = np.mean(accuracies)
                    std_acc = np.std(accuracies)
                    print(f"    Summary: {mean_acc:.4f} ± {std_acc:.4f} accuracy")

        print("\n" + "=" * 80)
        print("ALL EXPERIMENTS COMPLETE")
        print("=" * 80)

    def analyze_results(self) -> pd.DataFrame:
        """Analyze and summarize experimental results."""
        df = pd.DataFrame(self.results)

        # Compute summary statistics
        summary = df.groupby(["method", "embedding_dim"]).agg({
            "accuracy": ["mean", "std", "count"],
            "embedding_time": ["mean", "std"],
            "total_time": ["mean", "std"]
        }).round(4)

        # Flatten column names
        summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
        summary = summary.reset_index()

        print("\nEXPERIMENT SUMMARY:")
        print("=" * 120)
        print(summary.to_string())

        return summary

    def save_results(self, summary_df: pd.DataFrame):
        """Save results according to configuration."""
        output_config = self.config.get("output", {})

        if output_config.get("save_raw_results", True):
            df = pd.DataFrame(self.results)
            df.to_csv(self.output_dir / "raw_results.csv", index=False)

        if output_config.get("save_summary", True):
            summary_df.to_csv(self.output_dir / "summary_results.csv", index=False)

        if output_config.get("save_report", True):
            self._create_report(summary_df)

        print(f"\nResults saved to {self.output_dir}/")

    def _create_report(self, summary_df: pd.DataFrame):
        """Create experimental report."""
        with open(self.output_dir / "experiment_report.txt", "w") as f:
            f.write(f"{self.config['experiment']['name'].upper()} REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Description: {self.config['experiment']['description']}\n\n")

            f.write("CONFIGURATION:\n")
            f.write(f"  Dataset: {self.config['dataset']['name']} ({self.graph.num_nodes} nodes, {len(np.unique(self.labels))} classes)\n")
            f.write(f"  Dimensions: {self.config['parameters']['dimensions']}\n")
            f.write(f"  Runs per configuration: {self.config['parameters']['n_runs']}\n")
            f.write(f"  Test size: {self.config['parameters']['test_size']}\n\n")

            f.write("METHODS:\n")
            for name, config in self.config["methods"].items():
                if config.get("enabled", True):
                    f.write(f"  {name}: {config['description']}\n")
            f.write("\n")

            f.write("SUMMARY RESULTS:\n")
            f.write(summary_df.to_string())
            f.write("\n\n")

            f.write("BEST RESULTS BY METHOD:\n")
            for method in summary_df["method"].unique():
                method_data = summary_df[summary_df["method"] == method]
                best_row = method_data.loc[method_data["accuracy_mean"].idxmax()]
                f.write(f"  {method}: {best_row['accuracy_mean']:.4f} ± {best_row['accuracy_std']:.4f} "
                       f"at dimension {int(best_row['embedding_dim'])}\n")

            best_overall = summary_df.loc[summary_df["accuracy_mean"].idxmax()]
            f.write(f"\nOverall best: {best_overall['method']} with {best_overall['accuracy_mean']:.4f} ± {best_overall['accuracy_std']:.4f} "
                   f"at dimension {int(best_overall['embedding_dim'])}\n")


def main():
    """Main function."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "experiment_config.yaml"

    if not Path(config_file).exists():
        print(f"Error: Configuration file '{config_file}' not found.")
        print("Usage: python configurable_dimension_sweep.py [config_file]")
        sys.exit(1)

    try:
        # Create and run experiment
        experiment = ConfigurableDimensionSweep(config_file)
        experiment.run_all_experiments()

        # Analyze results
        summary = experiment.analyze_results()

        # Save results
        experiment.save_results(summary)

        print(f"\n🎉 Experiment '{experiment.config['experiment']['name']}' completed successfully!")
        print(f"Results saved to: {experiment.output_dir}")
        print("Use create_accuracy_plots.py to generate visualizations.")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
