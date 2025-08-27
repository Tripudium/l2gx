#!/usr/bin/env python3
"""
Create Accuracy Plots from Dimension Sweep Results

Standalone script for generating accuracy vs dimension plots from experimental results.
Supports multiple result directories and flexible plotting options.

Usage:
    python create_accuracy_plots.py [results_dir] [--config config.yaml]
    python create_accuracy_plots.py dimension_sweep_results
    python create_accuracy_plots.py results1 results2 results3 --compare

Examples:
    # Plot single experiment
    python create_accuracy_plots.py dimension_sweep_results
    
    # Compare multiple experiments
    python create_accuracy_plots.py cora_results pubmed_results --compare
    
    # Custom plotting configuration
    python create_accuracy_plots.py results --config plot_config.yaml
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from typing import List, Dict, Optional

# Configure matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")


class AccuracyPlotter:
    """Creates accuracy vs dimension plots from experiment results."""
    
    def __init__(self, results_dirs: List[str], plot_config: Optional[Dict] = None):
        """
        Initialize plotter with result directories.
        
        Args:
            results_dirs: List of result directory paths
            plot_config: Optional plotting configuration
        """
        self.results_dirs = [Path(d) for d in results_dirs]
        self.plot_config = plot_config or self._default_plot_config()
        self.results_data = {}
        
    def _default_plot_config(self) -> Dict:
        """Default plotting configuration."""
        return {
            "figure_size": [15, 12],
            "dpi": 300,
            "save_formats": ["pdf", "png"],
            "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "line_width": 2,
            "marker_size": 6,
            "error_capsize": 4,
            "grid_alpha": 0.3,
            "legend_fontsize": 10,
            "title_fontsize": 14,
            "label_fontsize": 12
        }
    
    def load_results(self):
        """Load results from all directories."""
        for results_dir in self.results_dirs:
            summary_file = results_dir / "summary_results.csv"
            
            if not summary_file.exists():
                print(f"Warning: {summary_file} not found, skipping {results_dir}")
                continue
            
            try:
                df = pd.read_csv(summary_file)
                
                # Try to get experiment name from config or use directory name
                config_file = results_dir / "experiment_config.yaml"
                if config_file.exists():
                    with open(config_file) as f:
                        config = yaml.safe_load(f)
                        exp_name = config.get("experiment", {}).get("name", results_dir.name)
                else:
                    exp_name = results_dir.name
                
                self.results_data[exp_name] = {
                    "data": df,
                    "path": results_dir
                }
                
                print(f"Loaded results from {results_dir}: {len(df)} configurations")
                
            except Exception as e:
                print(f"Error loading {summary_file}: {e}")
    
    def create_single_experiment_plot(self, exp_name: str, data: pd.DataFrame, output_dir: Path):
        """Create plots for a single experiment."""
        # Set up the plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.plot_config["figure_size"])
        
        methods = data["method"].unique()
        colors = self.plot_config["colors"][:len(methods)]
        
        # Plot 1: Accuracy vs Dimension
        for i, method in enumerate(methods):
            method_data = data[data["method"] == method]
            
            # Clean method name for display
            display_names = {
                "full_graph": "Full Graph",
                "l2g_rademacher": "L2G + Rademacher",
                "l2g_standard": "L2G Standard",
                "hierarchical_l2g": "Hierarchical + L2G",
                "geo_rademacher": "Geo + Rademacher"
            }
            display_name = display_names.get(method, method.replace("_", " ").title())
            
            ax1.errorbar(
                method_data["embedding_dim"], 
                method_data["accuracy_mean"],
                yerr=method_data["accuracy_std"],
                marker='o', linewidth=self.plot_config["line_width"], 
                markersize=self.plot_config["marker_size"],
                label=display_name, color=colors[i],
                capsize=self.plot_config["error_capsize"], capthick=1
            )
        
        ax1.set_xlabel("Embedding Dimension", fontsize=self.plot_config["label_fontsize"])
        ax1.set_ylabel("Classification Accuracy", fontsize=self.plot_config["label_fontsize"])
        ax1.set_title("Classification Accuracy vs Embedding Dimension", fontsize=self.plot_config["title_fontsize"])
        ax1.legend(fontsize=self.plot_config["legend_fontsize"])
        ax1.grid(True, alpha=self.plot_config["grid_alpha"])
        ax1.set_xscale("log", base=2)
        
        # Get unique dimensions and set them as ticks
        dimensions = sorted(data["embedding_dim"].unique())
        ax1.set_xticks(dimensions)
        ax1.set_xticklabels(dimensions)
        
        # Plot 2: Embedding Time vs Dimension
        for i, method in enumerate(methods):
            method_data = data[data["method"] == method]
            display_names_local = {
                "full_graph": "Full Graph",
                "l2g_rademacher": "L2G + Rademacher", 
                "l2g_standard": "L2G Standard",
                "hierarchical_l2g": "Hierarchical + L2G",
                "geo_rademacher": "Geo + Rademacher"
            }
            display_name = display_names_local.get(method, method.replace("_", " ").title())
            
            ax2.errorbar(
                method_data["embedding_dim"],
                method_data["embedding_time_mean"],
                yerr=method_data["embedding_time_std"],
                marker='s', linewidth=self.plot_config["line_width"],
                markersize=self.plot_config["marker_size"],
                label=display_name, color=colors[i],
                capsize=self.plot_config["error_capsize"], capthick=1
            )
        
        ax2.set_xlabel("Embedding Dimension", fontsize=self.plot_config["label_fontsize"])
        ax2.set_ylabel("Embedding Time (seconds)", fontsize=self.plot_config["label_fontsize"])
        ax2.set_title("Embedding Time vs Dimension", fontsize=self.plot_config["title_fontsize"])
        ax2.legend(fontsize=self.plot_config["legend_fontsize"])
        ax2.grid(True, alpha=self.plot_config["grid_alpha"])
        ax2.set_xscale("log", base=2)
        ax2.set_xticks(dimensions)
        ax2.set_xticklabels(dimensions)
        
        # Plot 3: Accuracy Heatmap
        pivot_acc = data.pivot(index="method", columns="embedding_dim", values="accuracy_mean")
        
        # Rename methods for display
        display_names_heatmap = {
            "full_graph": "Full Graph",
            "l2g_rademacher": "L2G + Rademacher", 
            "l2g_standard": "L2G Standard",
            "hierarchical_l2g": "Hierarchical + L2G",
            "geo_rademacher": "Geo + Rademacher"
        }
        pivot_acc.index = [display_names_heatmap.get(idx, idx.replace("_", " ").title()) for idx in pivot_acc.index]
        
        sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax3, 
                   cbar_kws={'label': 'Accuracy'})
        ax3.set_title("Accuracy Heatmap", fontsize=self.plot_config["title_fontsize"])
        ax3.set_xlabel("Embedding Dimension", fontsize=self.plot_config["label_fontsize"])
        ax3.set_ylabel("Method", fontsize=self.plot_config["label_fontsize"])
        
        # Plot 4: Accuracy vs Time Trade-off (for highest dimension)
        max_dim = data["embedding_dim"].max()
        max_dim_data = data[data["embedding_dim"] == max_dim]
        
        for i, method in enumerate(methods):
            method_data = max_dim_data[max_dim_data["method"] == method]
            display_names_scatter = {
                "full_graph": "Full Graph",
                "l2g_rademacher": "L2G + Rademacher", 
                "l2g_standard": "L2G Standard",
                "hierarchical_l2g": "Hierarchical + L2G",
                "geo_rademacher": "Geo + Rademacher"
            }
            display_name = display_names_scatter.get(method, method.replace("_", " ").title())
            
            if len(method_data) > 0:
                ax4.scatter(
                    method_data["embedding_time_mean"],
                    method_data["accuracy_mean"],
                    s=100, alpha=0.7, label=display_name,
                    color=colors[i]
                )
                
                # Add method label
                ax4.annotate(display_name,
                           (method_data["embedding_time_mean"].iloc[0], 
                            method_data["accuracy_mean"].iloc[0]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
        
        ax4.set_xlabel("Embedding Time (seconds)", fontsize=self.plot_config["label_fontsize"])
        ax4.set_ylabel("Classification Accuracy", fontsize=self.plot_config["label_fontsize"])
        ax4.set_title(f"Accuracy vs Time Trade-off (Dimension {max_dim})", fontsize=self.plot_config["title_fontsize"])
        ax4.grid(True, alpha=self.plot_config["grid_alpha"])
        
        plt.tight_layout()
        
        # Save plots
        for fmt in self.plot_config["save_formats"]:
            output_file = output_dir / f"{exp_name}_accuracy_plots.{fmt}"
            fig.savefig(output_file, format=fmt, dpi=self.plot_config["dpi"], bbox_inches='tight')
            print(f"Saved plot: {output_file}")
        
        return fig
    
    def create_comparison_plot(self, output_dir: Path):
        """Create comparison plot across multiple experiments."""
        if len(self.results_data) < 2:
            print("Need at least 2 experiments for comparison plot")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        colors = self.plot_config["colors"]
        color_idx = 0
        
        for exp_name, exp_data in self.results_data.items():
            data = exp_data["data"]
            methods = data["method"].unique()
            
            for method in methods:
                method_data = data[data["method"] == method]
                
                # Create combined label
                display_names = {
                    "full_graph": "Full Graph",
                    "l2g_rademacher": "L2G + Rademacher",
                    "l2g_standard": "L2G Standard",
                    "hierarchical_l2g": "Hierarchical + L2G",
                    "geo_rademacher": "Geo + Rademacher"
                }
                method_display = display_names.get(method, method.replace("_", " ").title())
                label = f"{exp_name} - {method_display}"
                
                ax.errorbar(
                    method_data["embedding_dim"],
                    method_data["accuracy_mean"],
                    yerr=method_data["accuracy_std"],
                    marker='o', linewidth=self.plot_config["line_width"],
                    markersize=self.plot_config["marker_size"],
                    label=label, color=colors[color_idx % len(colors)],
                    capsize=self.plot_config["error_capsize"], capthick=1
                )
                
                color_idx += 1
        
        ax.set_xlabel("Embedding Dimension", fontsize=self.plot_config["label_fontsize"])
        ax.set_ylabel("Classification Accuracy", fontsize=self.plot_config["label_fontsize"])
        ax.set_title("Accuracy Comparison Across Experiments", fontsize=self.plot_config["title_fontsize"])
        ax.legend(fontsize=self.plot_config["legend_fontsize"], bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=self.plot_config["grid_alpha"])
        ax.set_xscale("log", base=2)
        
        # Set dimension ticks
        all_dims = set()
        for exp_data in self.results_data.values():
            all_dims.update(exp_data["data"]["embedding_dim"].unique())
        dimensions = sorted(all_dims)
        ax.set_xticks(dimensions)
        ax.set_xticklabels(dimensions)
        
        plt.tight_layout()
        
        # Save comparison plot
        for fmt in self.plot_config["save_formats"]:
            output_file = output_dir / f"experiment_comparison.{fmt}"
            fig.savefig(output_file, format=fmt, dpi=self.plot_config["dpi"], bbox_inches='tight')
            print(f"Saved comparison plot: {output_file}")
        
        return fig
    
    def create_all_plots(self):
        """Create all plots based on loaded data."""
        if not self.results_data:
            print("No results data loaded. Run load_results() first.")
            return
        
        # Create individual experiment plots
        for exp_name, exp_data in self.results_data.items():
            print(f"\nCreating plots for {exp_name}...")
            self.create_single_experiment_plot(exp_name, exp_data["data"], exp_data["path"])
        
        # Create comparison plot if multiple experiments
        if len(self.results_data) > 1:
            print(f"\nCreating comparison plot...")
            # Use first experiment's directory for comparison plot
            first_dir = next(iter(self.results_data.values()))["path"]
            self.create_comparison_plot(first_dir)
        
        # Show plots
        plt.show()


def load_plot_config(config_path: str) -> Dict:
    """Load plotting configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Warning: Plot config {config_path} not found, using defaults")
        return {}
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    return config.get("plotting", {})


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Create accuracy plots from dimension sweep experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_accuracy_plots.py dimension_sweep_results
  python create_accuracy_plots.py cora_results pubmed_results --compare
  python create_accuracy_plots.py results --config plot_config.yaml
        """
    )
    
    parser.add_argument(
        "results_dirs", 
        nargs="+",
        help="One or more result directories to plot"
    )
    
    parser.add_argument(
        "--config",
        help="YAML configuration file for plotting options"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Create comparison plot across experiments"
    )
    
    args = parser.parse_args()
    
    # Validate result directories
    valid_dirs = []
    for results_dir in args.results_dirs:
        if Path(results_dir).exists():
            valid_dirs.append(results_dir)
        else:
            print(f"Warning: Directory {results_dir} not found, skipping")
    
    if not valid_dirs:
        print("Error: No valid result directories found")
        sys.exit(1)
    
    try:
        # Load plot configuration if provided
        plot_config = {}
        if args.config:
            plot_config = load_plot_config(args.config)
        
        # Create plotter and load results
        plotter = AccuracyPlotter(valid_dirs, plot_config)
        plotter.load_results()
        
        if not plotter.results_data:
            print("Error: No valid results data found")
            sys.exit(1)
        
        # Create plots
        plotter.create_all_plots()
        
        print(f"\n🎉 Plots created successfully!")
        print(f"Processed {len(plotter.results_data)} experiments")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()