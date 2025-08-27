#!/usr/bin/env python3
"""
Create classification accuracy plots from experiment results.
Shows classification accuracy vs embedding dimension for a single dataset.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class ExperimentPlotter:
    def __init__(self):
        self.plot_config = {
            "figure_size": (10, 6),
            "dpi": 300,
            "line_width": 2.5,
            "marker_size": 8,
            "error_capsize": 4,
            "title_fontsize": 16,
            "label_fontsize": 12,
            "legend_fontsize": 10,
            "grid_alpha": 0.3
        }

        self.colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]  # Blue, Orange, Green, Red, Purple

        self.display_names = {
            "full_graph": "Full Graph",
            "l2g_rademacher": "L2G + Rademacher",
            "l2g_standard": "L2G Standard",
            "hierarchical_l2g": "Hierarchical + L2G",
            "geo_rademacher": "Geo + Rademacher"
        }

    def load_experiment_data(self, results_dir: Path):
        """Load experiment data from results directory."""
        summary_file = results_dir / "summary_results.csv"
        if not summary_file.exists():
            raise FileNotFoundError(f"Summary results not found: {summary_file}")

        data = pd.read_csv(summary_file)
        return data

    def create_plot(self, data: pd.DataFrame, dataset_name: str = None):
        """Create accuracy plot for a single dataset."""
        fig, ax = plt.subplots(1, 1, figsize=self.plot_config["figure_size"])

        methods = sorted(data["method"].unique())
        dimensions = sorted(data["embedding_dim"].unique())

        for i, method in enumerate(methods):
            method_data = data[data["method"] == method]
            display_name = self.display_names.get(method, method.replace("_", " ").title())

            ax.errorbar(
                method_data["embedding_dim"],
                method_data["accuracy_mean"],
                yerr=method_data["accuracy_std"],
                marker='o',
                linewidth=self.plot_config["line_width"],
                markersize=self.plot_config["marker_size"],
                label=display_name,
                color=self.colors[i % len(self.colors)],
                capsize=self.plot_config["error_capsize"],
                capthick=1
            )

        ax.set_xlabel("Embedding Dimension", fontsize=self.plot_config["label_fontsize"])
        ax.set_ylabel("Classification Accuracy", fontsize=self.plot_config["label_fontsize"])

        # Set title based on dataset name or infer from directory
        if dataset_name:
            title = dataset_name
        else:
            title = "Classification Accuracy vs Embedding Dimension"

        ax.set_title(title, fontsize=self.plot_config["title_fontsize"])
        ax.grid(True, alpha=self.plot_config["grid_alpha"])
        ax.set_xscale("log", base=2)
        ax.set_xticks(dimensions)
        ax.set_xticklabels(dimensions)
        ax.legend(fontsize=self.plot_config["legend_fontsize"])

        # Set y-axis limits for better comparison
        ax.set_ylim(0.3, 0.9)

        plt.tight_layout()
        return fig

    def save_plots(self, fig, output_dir: Path, filename: str):
        """Save plots in PDF and PNG formats."""
        output_dir.mkdir(exist_ok=True)

        # Save PDF
        pdf_path = output_dir / f"{filename}.pdf"
        fig.savefig(pdf_path, dpi=self.plot_config["dpi"], bbox_inches='tight')
        print(f"Saved PDF: {pdf_path}")

        # Save PNG
        png_path = output_dir / f"{filename}.png"
        fig.savefig(png_path, dpi=self.plot_config["dpi"], bbox_inches='tight')
        print(f"Saved PNG: {png_path}")

def main():
    parser = argparse.ArgumentParser(description="Create classification accuracy plots from experiment results")
    parser.add_argument("results_dir", type=Path, help="Path to results directory containing summary_results.csv")
    parser.add_argument("--output_dir", type=Path, default=Path("plots"),
                       help="Output directory for plots (default: plots)")
    parser.add_argument("--output_name", type=str, default=None,
                       help="Output filename (without extension). Default: inferred from results_dir name")
    parser.add_argument("--title", type=str, default=None,
                       help="Plot title. Default: inferred from results directory name")

    args = parser.parse_args()

    # Validate results directory
    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)

    # Load data
    plotter = ExperimentPlotter()

    try:
        data = plotter.load_experiment_data(args.results_dir)
        print(f"Loaded data from {args.results_dir}: {len(data)} rows")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Infer dataset name from directory if not provided
    if args.title:
        dataset_name = args.title
    else:
        dir_name = args.results_dir.name.lower()
        if "cora" in dir_name:
            dataset_name = "Cora"
        elif "pubmed" in dir_name:
            dataset_name = "PubMed"
        elif "citeseer" in dir_name:
            dataset_name = "CiteSeer"
        else:
            dataset_name = args.results_dir.name.replace("_", " ").title()

    fig = plotter.create_plot(data, dataset_name)

    if args.output_name:
        filename = args.output_name
    else:
        filename = args.results_dir.name
        if filename.endswith("_results"):
            filename = filename[:-8]  # Remove "_results" suffix
        filename = f"{filename}_plot"

    plotter.save_plots(fig, args.output_dir, filename)

    print("\nExperiment Summary:")
    print(f"Results directory: {args.results_dir}")
    print(f"Dataset: {dataset_name}")
    print(f"Plots saved to: {args.output_dir}/{filename}.pdf and {args.output_dir}/{filename}.png")

if __name__ == "__main__":
    main()
