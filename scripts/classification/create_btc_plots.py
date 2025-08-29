#!/usr/bin/env python3
"""
Create classification accuracy plots for BTC-reduced experiment results.
Shows classification accuracy vs embedding dimension with proper method names.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class BTCExperimentPlotter:
    def __init__(self):
        self.plot_config = {
            "figure_size": (12, 7),
            "dpi": 300,
            "line_width": 2.5,
            "marker_size": 8,
            "error_capsize": 4,
            "title_fontsize": 16,
            "label_fontsize": 14,
            "legend_fontsize": 11,
            "grid_alpha": 0.3
        }

        # Extended color palette for more methods
        self.colors = [
            "#1f77b4",  # Blue
            "#ff7f0e",  # Orange
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#9467bd",  # Purple
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
        ]

        # Updated display names including all methods
        self.display_names = {
            "full_graph": "Full Graph VGAE",
            "l2g_rademacher": "L2G + Rademacher",
            "l2g_standard": "L2G Standard",
            "hierarchical_l2g": "Hierarchical + L2G",
            "geo_rademacher": "Geo + Rademacher",
            "graphsage_full": "GraphSAGE (Full)",
            "patched_l2g_rademacher": "Patched L2G + Rademacher",
            "patched_geo": "Patched Geo",
            "hierarchical_unified": "Hierarchical (Unified)",
        }

        # Marker styles for different methods
        self.markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']

    def load_experiment_data(self, results_dir: Path):
        """Load experiment data from results directory."""
        summary_file = results_dir / "summary_results.csv"
        if not summary_file.exists():
            # Try looking for raw results
            raw_file = results_dir / "raw_results.csv"
            if raw_file.exists():
                print(f"Using raw results file: {raw_file}")
                raw_data = pd.read_csv(raw_file)
                # Create summary from raw data
                summary = raw_data.groupby(['method', 'embedding_dim'])['accuracy'].agg([
                    ('accuracy_mean', 'mean'),
                    ('accuracy_std', 'std'),
                    ('accuracy_count', 'count')
                ]).reset_index()
                return summary
            else:
                raise FileNotFoundError(f"Neither summary nor raw results found in: {results_dir}")
        
        data = pd.read_csv(summary_file)
        return data

    def create_plot(self, data: pd.DataFrame, dataset_name: str = None):
        """Create accuracy plot for BTC-reduced dataset."""
        fig, ax = plt.subplots(1, 1, figsize=self.plot_config["figure_size"])

        methods = sorted(data["method"].unique())
        dimensions = sorted(data["embedding_dim"].unique())

        # Track best accuracy for annotation
        best_accuracy = 0
        best_method = ""
        best_dim = 0

        for i, method in enumerate(methods):
            method_data = data[data["method"] == method].sort_values("embedding_dim")
            display_name = self.display_names.get(method, method.replace("_", " ").title())

            # Get color and marker
            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]

            # Plot with error bars
            if "accuracy_std" in method_data.columns:
                ax.errorbar(
                    method_data["embedding_dim"],
                    method_data["accuracy_mean"],
                    yerr=method_data["accuracy_std"],
                    marker=marker,
                    linewidth=self.plot_config["line_width"],
                    markersize=self.plot_config["marker_size"],
                    label=display_name,
                    color=color,
                    capsize=self.plot_config["error_capsize"],
                    capthick=1,
                    alpha=0.9
                )
                accuracies = method_data["accuracy_mean"].values
            else:
                # No std deviation available
                ax.plot(
                    method_data["embedding_dim"],
                    method_data["accuracy_mean"],
                    marker=marker,
                    linewidth=self.plot_config["line_width"],
                    markersize=self.plot_config["marker_size"],
                    label=display_name,
                    color=color,
                    alpha=0.9
                )
                accuracies = method_data["accuracy_mean"].values

            # Track best performance
            max_acc = np.max(accuracies)
            if max_acc > best_accuracy:
                best_accuracy = max_acc
                best_method = display_name
                best_dim = method_data.loc[method_data["accuracy_mean"].idxmax(), "embedding_dim"]

        # Styling
        ax.set_xlabel("Embedding Dimension", fontsize=self.plot_config["label_fontsize"])
        ax.set_ylabel("Classification Accuracy", fontsize=self.plot_config["label_fontsize"])

        # Set title
        if dataset_name:
            title = f"{dataset_name} - Classification Accuracy vs Embedding Dimension"
        else:
            title = "BTC-Reduced Entity Classification: Accuracy vs Embedding Dimension"

        ax.set_title(title, fontsize=self.plot_config["title_fontsize"], pad=20)
        
        # Grid
        ax.grid(True, alpha=self.plot_config["grid_alpha"], linestyle='--')
        
        # X-axis log scale
        ax.set_xscale("log", base=2)
        ax.set_xticks(dimensions)
        ax.set_xticklabels(dimensions)
        
        # Legend
        ax.legend(
            fontsize=self.plot_config["legend_fontsize"],
            loc='best',
            framealpha=0.95,
            edgecolor='gray'
        )

        # Y-axis limits - adjust based on data
        min_acc = data["accuracy_mean"].min()
        max_acc = data["accuracy_mean"].max()
        y_margin = (max_acc - min_acc) * 0.1
        ax.set_ylim(max(0, min_acc - y_margin), min(1.0, max_acc + y_margin))

        # Add best result annotation
        ax.annotate(
            f'Best: {best_method}\n{best_accuracy:.3f} @ dim={best_dim}',
            xy=(0.02, 0.98),
            xycoords='axes fraction',
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3)
        )

        # Add class count info for BTC
        ax.text(
            0.98, 0.02,
            "11 Bitcoin Entity Classes",
            transform=ax.transAxes,
            fontsize=10,
            ha='right',
            va='bottom',
            style='italic',
            alpha=0.7
        )

        plt.tight_layout()
        return fig

    def save_plots(self, fig, output_dir: Path, filename: str):
        """Save plots in PDF and PNG formats."""
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save PDF
        pdf_path = output_dir / f"{filename}.pdf"
        fig.savefig(pdf_path, dpi=self.plot_config["dpi"], bbox_inches='tight')
        print(f"✓ Saved PDF: {pdf_path}")

        # Save PNG
        png_path = output_dir / f"{filename}.png"
        fig.savefig(png_path, dpi=self.plot_config["dpi"], bbox_inches='tight')
        print(f"✓ Saved PNG: {png_path}")

    def print_summary(self, data: pd.DataFrame):
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        # Best overall result
        best_idx = data["accuracy_mean"].idxmax()
        best_row = data.loc[best_idx]
        print(f"\nBest Result:")
        print(f"  Method: {self.display_names.get(best_row['method'], best_row['method'])}")
        print(f"  Dimension: {best_row['embedding_dim']}")
        print(f"  Accuracy: {best_row['accuracy_mean']:.4f}", end="")
        if "accuracy_std" in data.columns and not pd.isna(best_row.get('accuracy_std')):
            print(f" ± {best_row['accuracy_std']:.4f}")
        else:
            print()
        
        # Per-method best
        print("\nBest per Method:")
        for method in sorted(data["method"].unique()):
            method_data = data[data["method"] == method]
            best_idx = method_data["accuracy_mean"].idxmax()
            best_row = method_data.loc[best_idx]
            display_name = self.display_names.get(method, method)
            print(f"  {display_name:30s}: {best_row['accuracy_mean']:.4f} @ dim={best_row['embedding_dim']}")


def main():
    parser = argparse.ArgumentParser(
        description="Create classification accuracy plots for BTC-reduced experiments"
    )
    parser.add_argument(
        "results_dir", 
        type=Path, 
        help="Path to results directory containing summary_results.csv or raw_results.csv"
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=Path("plots"),
        help="Output directory for plots (default: plots)"
    )
    parser.add_argument(
        "--output_name", 
        type=str, 
        default=None,
        help="Output filename (without extension). Default: btc_reduced_results"
    )
    parser.add_argument(
        "--title", 
        type=str, 
        default="BTC-Reduced (Bitcoin Entity Classification)",
        help="Plot title"
    )

    args = parser.parse_args()

    # Validate results directory
    if not args.results_dir.exists():
        print(f"❌ Error: Results directory not found: {args.results_dir}")
        sys.exit(1)

    # Create plotter
    plotter = BTCExperimentPlotter()

    # Load data
    try:
        data = plotter.load_experiment_data(args.results_dir)
        print(f"✓ Loaded data from {args.results_dir}")
        print(f"  Found {len(data)} method-dimension combinations")
        print(f"  Methods: {', '.join(sorted(data['method'].unique()))}")
        print(f"  Dimensions: {sorted(data['embedding_dim'].unique())}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

    # Create plot
    fig = plotter.create_plot(data, args.title)

    # Determine output filename
    if args.output_name:
        filename = args.output_name
    else:
        filename = "btc_reduced_classification_results"

    # Save plots
    plotter.save_plots(fig, args.output_dir, filename)

    # Print summary
    plotter.print_summary(data)

    print(f"\n✅ Plots saved to: {args.output_dir}/")
    print(f"   - {filename}.pdf")
    print(f"   - {filename}.png")


if __name__ == "__main__":
    main()