#!/usr/bin/env python3
"""
Plot noise robustness results with error bars from CSV data.

This script reads the noise_robustness_results.csv file and creates a plot showing
Procrustes error vs noise level for three dimensions (32, 64, 128) with error bars.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_synthetic_results(csv_file: str, output_file: str):
    """
    Create plot of noise robustness results with error bars.

    Args:
        csv_file: Path to the CSV file containing noise robustness results
        output_file: Path where the output PDF should be saved
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Define colors for each dimension
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green

    # Plot each dimension with error bars
    for i, dim in enumerate([32, 64, 128]):
        dim_data = df[df["dimension"] == dim]

        plt.errorbar(
            dim_data["noise_level"],
            dim_data["avg_procrustes_error"],
            yerr=dim_data["std_procrustes_error"],
            marker="o",
            linewidth=2,
            markersize=6,
            capsize=4,
            capthick=1.5,
            color=colors[i],
            label=f"{dim}",
        )

    # set labels
    plt.xlabel("Noise")
    plt.ylabel("Procrustes error")

    # Add legend
    plt.legend()

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Ensure tight layout
    plt.tight_layout()

    # Save as PDF
    plt.savefig(output_file, format="pdf", bbox_inches="tight")

    print(f"Plot saved as {output_file}")

    # Display statistics about the error bars
    print("\nError bar statistics:")
    for dim in [32, 64, 128]:
        dim_data = df[df["dimension"] == dim]
        if not dim_data.empty:
            mean_std = dim_data["std_procrustes_error"].mean()
            max_std = dim_data["std_procrustes_error"].max()
            print(
                f"Dimension {dim}: mean std = {mean_std:.6f}, max std = {max_std:.6f}"
            )


def main():
    """Main function to handle command line arguments and run the plotting."""
    parser = argparse.ArgumentParser(
        description="Plot noise robustness results with error bars"
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the CSV file containing noise robustness results",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="noise_robustness_plot.pdf",
        help="Output filename for the PDF plot (default: noise_robustness_plot.pdf)",
    )

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.csv_file).exists():
        print(f"Error: Input file '{args.csv_file}' not found!")
        return 1

    # Create the plot
    plot_synthetic_results(args.csv_file, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
