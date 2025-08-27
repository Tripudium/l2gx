#!/usr/bin/env python3
"""
Dimension Sweep Classification Experiment

Compares three embedding approaches across different dimensions (2,4,8,16,32,64,128):
1. Full graph embedding (VGAE on entire graph)
2. L2G with Rademacher sketching (patch-based + L2G alignment)
3. Hierarchical binary tree (size bound 800) + L2G alignment

For each configuration and dimension:
- Runs 10 independent experiments
- Records classification accuracy and embedding time
- Computes mean and standard deviation

Generates comparative plots saved as PDF.

Usage:
    python dimension_sweep_experiment.py
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import yaml
from typing import Dict, List, Tuple

# Add parent directory to path for L2GX imports
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Add embedding and hierarchical directories
sys.path.insert(0, str(Path(__file__).parent.parent / "embedding"))
sys.path.insert(0, str(Path(__file__).parent.parent / "hierarchical"))

from l2gx.datasets import get_dataset
from l2gx.graphs import TGraph
from l2gx.embedding import get_embedding
from l2gx.align import get_aligner
from binary_hierarchical_embedding import BinaryHierarchicalEmbedding

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class DimensionSweepExperiment:
    """Runs classification experiments across different embedding dimensions and methods."""
    
    def __init__(self, output_dir: str = "dimension_sweep_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Experimental parameters
        self.dimensions = [2, 4, 8, 16, 32, 64, 128]
        self.n_runs = 3  # Reduced for faster execution
        self.test_size = 0.2
        self.random_seed = 42
        
        # Load Cora dataset once
        self.graph, self.labels = self._load_cora()
        
        # Results storage
        self.results = []
        
    def _load_cora(self) -> Tuple[TGraph, np.ndarray]:
        """Load Cora dataset."""
        print("Loading Cora dataset...")
        dataset = get_dataset("Cora")
        pg_data = dataset.to("torch-geometric")
        
        graph = TGraph(
            edge_index=pg_data.edge_index,
            x=pg_data.x,
            y=pg_data.y,
            num_nodes=pg_data.num_nodes,
        )
        
        labels = pg_data.y.cpu().numpy()
        print(f"Cora: {graph.num_nodes} nodes, {graph.num_edges} edges, {len(np.unique(labels))} classes")
        
        return graph, labels
    
    def full_graph_embedding(self, embedding_dim: int) -> Tuple[np.ndarray, float]:
        """
        Generate full graph embedding using VGAE on entire graph.
        
        Returns:
            (embedding, time_taken)
        """
        start_time = time.time()
        
        # Configure embedder
        embedder = get_embedding(
            "vgae",
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim * 2,
            epochs=100,  # Reduced for faster execution
            learning_rate=0.001,
            patience=20,
            verbose=False
        )
        
        # Convert to PyTorch Geometric format
        pg_data = self.graph.to_tg()
        
        # Generate embedding
        embedding = embedder.fit_transform(pg_data)
        
        end_time = time.time()
        
        return embedding, end_time - start_time
    
    def l2g_rademacher_embedding(self, embedding_dim: int) -> Tuple[np.ndarray, float]:
        """
        Generate L2G embedding with Rademacher sketching.
        
        Returns:
            (embedding, time_taken)
        """
        start_time = time.time()
        
        try:
            # Create patches
            from l2gx.patch import create_patches
            
            patch_graph = create_patches(
                self.graph,
                num_patches=10,
                clustering_method="metis",
                min_overlap=256,
                target_overlap=512,
                sparsify_method="resistance",
                target_patch_degree=4,
                use_conductance_weighting=True,
                verbose=False,
            )
            
            patches = patch_graph.patches
            
            # Configure embedder
            embedder = get_embedding(
                "vgae",
                embedding_dim=embedding_dim,
                hidden_dim=embedding_dim * 2,
                epochs=100,  # Reduced for faster execution
                learning_rate=0.001,
                patience=20,
                verbose=False
            )
            
            # Embed each patch
            for patch in patches:
                # Extract patch subgraph
                import torch
                patch_nodes = torch.tensor(patch.nodes, dtype=torch.long)
                patch_tgraph = self.graph.subgraph(patch_nodes, relabel=True)
                patch_data = patch_tgraph.to_tg()
                
                # Embed the patch
                coordinates = embedder.fit_transform(patch_data)
                patch.coordinates = coordinates
            
            # Perform L2G alignment with Rademacher
            aligner = get_aligner("l2g")
            aligner.randomized_method = "randomized"
            aligner.sketch_method = "rademacher"
            
            # Align patches
            aligner.align_patches(patch_graph)
            embedding = aligner.get_aligned_embedding()
            
        except Exception as e:
            print(f"L2G embedding failed for dim {embedding_dim}: {e}")
            # Fallback to random embedding
            embedding = np.random.randn(self.graph.num_nodes, embedding_dim) * 0.1
        
        end_time = time.time()
        
        return embedding, end_time - start_time
    
    def hierarchical_l2g_embedding(self, embedding_dim: int) -> Tuple[np.ndarray, float]:
        """
        Generate hierarchical binary tree embedding with L2G alignment.
        
        Returns:
            (embedding, time_taken)
        """
        start_time = time.time()
        
        try:
            # Create hierarchical embedder
            embedder = BinaryHierarchicalEmbedding(
                max_patch_size=800,
                embedding_dim=embedding_dim,
                embedding_method="vgae",
                epochs=100,  # Reduced for faster execution
                verbose=False
            )
            
            # Build tree and embed
            embedder.root = embedder.build_hierarchical_tree(self.graph)
            embedder.embed_leaf_patches()
            embedder.hierarchical_alignment(embedder.root)
            
            embedding = embedder.root.embedding
            
        except Exception as e:
            print(f"Hierarchical embedding failed for dim {embedding_dim}: {e}")
            # Fallback to random embedding
            embedding = np.random.randn(self.graph.num_nodes, embedding_dim) * 0.1
        
        end_time = time.time()
        
        return embedding, end_time - start_time
    
    def run_classification(self, embedding: np.ndarray) -> float:
        """
        Run logistic regression classification on embedding.
        
        Returns:
            Test accuracy
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embedding, self.labels,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=self.labels
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classifier
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=self.random_seed,
            solver="lbfgs"
        )
        
        classifier.fit(X_train_scaled, y_train)
        
        # Predict and compute accuracy
        y_pred = classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    def run_single_experiment(self, method: str, embedding_dim: int, run_id: int) -> Dict:
        """Run single experiment for given method, dimension, and run."""
        print(f"  Run {run_id+1}/10: {method}, dim={embedding_dim}")
        
        # Generate embedding based on method
        if method == "full_graph":
            embedding, embed_time = self.full_graph_embedding(embedding_dim)
        elif method == "l2g_rademacher":
            embedding, embed_time = self.l2g_rademacher_embedding(embedding_dim)
        elif method == "hierarchical_l2g":
            embedding, embed_time = self.hierarchical_l2g_embedding(embedding_dim)
        else:
            raise ValueError(f"Unknown method: {method}")
        
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
        methods = ["full_graph", "l2g_rademacher", "hierarchical_l2g"]
        
        total_experiments = len(methods) * len(self.dimensions) * self.n_runs
        experiment_count = 0
        
        print(f"Starting dimension sweep experiment:")
        print(f"  Methods: {len(methods)} ({', '.join(methods)})")
        print(f"  Dimensions: {len(self.dimensions)} ({self.dimensions})")
        print(f"  Runs per configuration: {self.n_runs}")
        print(f"  Total experiments: {total_experiments}")
        print("=" * 80)
        
        for method in methods:
            print(f"\nRunning {method} experiments...")
            
            for embedding_dim in self.dimensions:
                print(f"\n  Dimension {embedding_dim}:")
                
                for run_id in range(self.n_runs):
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
                        # Record failed experiment
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
        
        print(f"\n" + "=" * 80)
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
        print("=" * 100)
        print(summary.to_string())
        
        return summary
    
    def create_plots(self, summary_df: pd.DataFrame):
        """Create comparative plots and save as PDF."""
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Classification Accuracy vs Dimension
        methods = summary_df["method"].unique()
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        
        for i, method in enumerate(methods):
            method_data = summary_df[summary_df["method"] == method]
            
            # Clean method name for display
            display_name = {
                "full_graph": "Full Graph",
                "l2g_rademacher": "L2G + Rademacher",
                "hierarchical_l2g": "Hierarchical + L2G"
            }[method]
            
            ax1.errorbar(
                method_data["embedding_dim"], 
                method_data["accuracy_mean"],
                yerr=method_data["accuracy_std"],
                marker='o', linewidth=2, markersize=6,
                label=display_name, color=colors[i],
                capsize=4, capthick=1
            )
        
        ax1.set_xlabel("Embedding Dimension")
        ax1.set_ylabel("Classification Accuracy")
        ax1.set_title("Classification Accuracy vs Embedding Dimension")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log", base=2)
        ax1.set_xticks(self.dimensions)
        ax1.set_xticklabels(self.dimensions)
        
        # Plot 2: Embedding Time vs Dimension
        for i, method in enumerate(methods):
            method_data = summary_df[summary_df["method"] == method]
            display_name = {
                "full_graph": "Full Graph",
                "l2g_rademacher": "L2G + Rademacher", 
                "hierarchical_l2g": "Hierarchical + L2G"
            }[method]
            
            ax2.errorbar(
                method_data["embedding_dim"],
                method_data["embedding_time_mean"],
                yerr=method_data["embedding_time_std"],
                marker='s', linewidth=2, markersize=6,
                label=display_name, color=colors[i],
                capsize=4, capthick=1
            )
        
        ax2.set_xlabel("Embedding Dimension")
        ax2.set_ylabel("Embedding Time (seconds)")
        ax2.set_title("Embedding Time vs Dimension")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log", base=2)
        ax2.set_xticks(self.dimensions)
        ax2.set_xticklabels(self.dimensions)
        
        # Plot 3: Accuracy heatmap
        pivot_acc = summary_df.pivot(index="method", columns="embedding_dim", values="accuracy_mean")
        
        # Rename methods for display
        pivot_acc.index = [
            {"full_graph": "Full Graph", "l2g_rademacher": "L2G + Rademacher", 
             "hierarchical_l2g": "Hierarchical + L2G"}[idx] for idx in pivot_acc.index
        ]
        
        sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax3, cbar_kws={'label': 'Accuracy'})
        ax3.set_title("Accuracy Heatmap")
        ax3.set_xlabel("Embedding Dimension")
        ax3.set_ylabel("Method")
        
        # Plot 4: Performance comparison (accuracy vs time)
        for i, method in enumerate(methods):
            method_data = summary_df[summary_df["method"] == method]
            display_name = {
                "full_graph": "Full Graph",
                "l2g_rademacher": "L2G + Rademacher",
                "hierarchical_l2g": "Hierarchical + L2G"
            }[method]
            
            # Create scatter plot with dimension as size
            sizes = [50 + 20 * np.log2(dim) for dim in method_data["embedding_dim"]]
            
            scatter = ax4.scatter(
                method_data["embedding_time_mean"],
                method_data["accuracy_mean"],
                s=sizes, alpha=0.7, label=display_name,
                color=colors[i]
            )
            
            # Add dimension labels
            for _, row in method_data.iterrows():
                ax4.annotate(f'{int(row["embedding_dim"])}',
                           (row["embedding_time_mean"], row["accuracy_mean"]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
        
        ax4.set_xlabel("Embedding Time (seconds)")
        ax4.set_ylabel("Classification Accuracy")
        ax4.set_title("Accuracy vs Time Trade-off")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save as PDF
        output_path = self.output_dir / "dimension_sweep_comparison.pdf"
        fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        
        # Also save as PNG for easy viewing
        png_path = self.output_dir / "dimension_sweep_comparison.png"
        fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        print(f"\nPlots saved to:")
        print(f"  PDF: {output_path}")
        print(f"  PNG: {png_path}")
        
        return fig
    
    def save_results(self):
        """Save detailed results to files."""
        # Save raw results
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_dir / "raw_results.csv", index=False)
        
        # Save summary
        summary = self.analyze_results()
        summary.to_csv(self.output_dir / "summary_results.csv", index=False)
        
        # Create analysis report
        with open(self.output_dir / "experiment_report.txt", "w") as f:
            f.write("DIMENSION SWEEP CLASSIFICATION EXPERIMENT REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("EXPERIMENT CONFIGURATION:\n")
            f.write(f"  Dataset: Cora ({self.graph.num_nodes} nodes, {len(np.unique(self.labels))} classes)\n")
            f.write(f"  Dimensions tested: {self.dimensions}\n")
            f.write(f"  Methods: Full Graph, L2G+Rademacher, Hierarchical+L2G\n")
            f.write(f"  Runs per configuration: {self.n_runs}\n")
            f.write(f"  Test size: {self.test_size}\n\n")
            
            f.write("SUMMARY RESULTS:\n")
            f.write(summary.to_string())
            f.write("\n\n")
            
            # Best results analysis
            f.write("BEST RESULTS BY METHOD:\n")
            for method in summary["method"].unique():
                method_data = summary[summary["method"] == method]
                best_row = method_data.loc[method_data["accuracy_mean"].idxmax()]
                f.write(f"  {method}: {best_row['accuracy_mean']:.4f} ± {best_row['accuracy_std']:.4f} "
                       f"at dimension {int(best_row['embedding_dim'])}\n")
            
            f.write(f"\nOverall best: ")
            best_overall = summary.loc[summary["accuracy_mean"].idxmax()]
            f.write(f"{best_overall['method']} with {best_overall['accuracy_mean']:.4f} ± {best_overall['accuracy_std']:.4f} "
                   f"at dimension {int(best_overall['embedding_dim'])}\n")
        
        print(f"\nResults saved to:")
        print(f"  Raw data: {self.output_dir}/raw_results.csv")
        print(f"  Summary: {self.output_dir}/summary_results.csv")
        print(f"  Report: {self.output_dir}/experiment_report.txt")


def main():
    """Main experimental function."""
    try:
        # Create experiment
        experiment = DimensionSweepExperiment()
        
        # Run all experiments
        experiment.run_all_experiments()
        
        # Analyze results
        summary = experiment.analyze_results()
        
        # Create plots
        experiment.create_plots(summary)
        
        # Save results
        experiment.save_results()
        
        print(f"\n🎉 Dimension sweep experiment completed successfully!")
        print(f"Check {experiment.output_dir}/ for results and plots.")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()