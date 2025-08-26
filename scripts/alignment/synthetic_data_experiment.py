#!/usr/bin/env python3
"""
Noise robustness on synthetic data.

Tests parametric geo with 1 epoch and sparse_aware initialization across:
- Noise levels: 0.0 to 0.3 in increments of 0.01
- Dimensions: 32, 64, 128
- 10 repetitions per configuration
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from data_generation import generate_patch_graph, generate_points, transform_patches
from tqdm import tqdm

from l2gx.align import get_aligner, procrustes_error

OUTPUT_DIR = Path("results")
noise_levels = np.arange(0.0, 0.31, 0.01)  # 0.0 to 0.3 in increments of 0.01
dimensions = [32, 64, 128]

def load_base_config(config_file="data_config.yaml"):
    """Load the base data configuration."""
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config


def run_experiment(noise_levels: np.ndarray = noise_levels,
                   dimensions: list[int] = dimensions,
                   n_repetitions: int = 10,
                   output_dir: Path = OUTPUT_DIR
                   ):
    """Run noise robustness experiment with synthetic data."""
    print("=" * 80)
    print("NOISE ROBUSTNESS")
    print("=" * 80)

    # Experiment parameters
    print(
        f"Noise levels: {len(noise_levels)} levels from {noise_levels[0]:.2f} to {noise_levels[-1]:.2f}"
    )
    print(f"Dimensions: {dimensions}")
    print(f"Repetitions per config: {n_repetitions}")
    print(f"Total experiments: {len(noise_levels) * len(dimensions) * n_repetitions}")

    # Load base configuration
    base_config = load_base_config()

    results = []

    for dim in dimensions:
        print(f"\n{'-' * 60}")
        print(f"TESTING DIMENSION {dim}")
        print(f"{'-' * 60}")

        config = copy.deepcopy(base_config)
        config["point_spec"]["dim"] = dim

        for noise_level in tqdm(noise_levels, desc=f"Noise levels (dim={dim})"):
            config["noise_profile"]["noise_level"] = float(noise_level)

            noise_errors = []

            for rep in range(n_repetitions):
                try:
                    # Generate fresh data for each repetition
                    points, transformed_pg = generate_test_data(
                        config, seed=rep * 1000 + int(noise_level * 100)
                    )

                    error = test_alignment(points, transformed_pg)
                    noise_errors.append(error)

                except Exception as e:
                    print(
                        f"\nError at noise={noise_level:.2f}, dim={dim}, rep={rep}: {e}"
                    )
                    noise_errors.append(float("inf"))

            # Store results for this noise level
            avg_error = np.mean([e for e in noise_errors if e != float("inf")])
            std_error = np.std([e for e in noise_errors if e != float("inf")])
            valid_runs = sum(1 for e in noise_errors if e != float("inf"))

            results.append(
                {
                    "dimension": dim,
                    "noise_level": noise_level,
                    "avg_procrustes_error": avg_error,
                    "std_procrustes_error": std_error,
                    "valid_runs": valid_runs,
                    "all_errors": noise_errors,
                }
            )

            if noise_level % 0.05 == 0:  # Print progress every 5th noise level
                print(
                    f"  Noise {noise_level:.2f}: avg_error={avg_error:.4f} (±{std_error:.4f}), {valid_runs}/{n_repetitions} valid"
                )

    df = pd.DataFrame(results)
    output_dir.mkdir(exist_ok=True)

    csv_file = output_dir / "noise_robustness_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nResults saved to {csv_file}")

    print_summary_statistics(df)

    return df


def generate_test_data(config, seed=42):
    """Generate test data with specified configuration."""
    np.random.seed(seed)

    # Generate points
    points = generate_points(**config["point_spec"])

    # Generate patch graph
    pg = generate_patch_graph(points, **config["patch_spec"])

    # Transform patches
    transformed_patches = [copy.deepcopy(p) for p in pg.patches]

    # Extract noise profile parameters
    noise_level = config["noise_profile"]["noise_level"]
    shift_scale = config["noise_profile"]["shift_scale"]
    scale_range = config["noise_profile"]["scale_range"]

    # Add noise to patches
    if noise_level > 0:
        for patch in transformed_patches:
            noise = np.random.normal(
                loc=0, scale=noise_level, size=patch.coordinates.shape
            )
            patch.coordinates += noise

    # Apply transformations
    transformed_patches = transform_patches(
        transformed_patches, shift_scale=shift_scale, scale_range=scale_range
    )

    # Create transformed patch graph
    transformed_pg = copy.deepcopy(pg)
    transformed_pg.patches = transformed_patches

    return points, transformed_pg


def test_alignment(points, transformed_pg):
    """Test parametric geo alignment with sparse_aware initialization."""
    # Create aligner with specified configuration
    aligner = get_aligner(
        "geo",
        method="orthogonal",
        use_randomized_init=True,
        randomized_method="randomized",
    )

    # Align patches with 1 epoch
    aligner.align_patches(
        patch_graph=copy.deepcopy(transformed_pg),
        use_scale=True,
        num_epochs=2,
        learning_rate=0.01,
    )

    embedding = aligner.get_aligned_embedding()

    error = procrustes_error(points, embedding)

    return error


def print_summary_statistics(df):
    """Print comprehensive summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Overall statistics
    print(f"Total experiments: {len(df)}")
    print(f"Overall mean error: {df['avg_procrustes_error'].mean():.4f}")
    print(f"Overall std error: {df['avg_procrustes_error'].std():.4f}")

    # Statistics by dimension
    print("\nStatistics by dimension:")
    print(
        f"{'Dim':<6} {'Mean Error':<12} {'Std Error':<12} {'Min Error':<12} {'Max Error':<12}"
    )
    print("-" * 60)

    for dim in sorted(df["dimension"].unique()):
        dim_data = df[df["dimension"] == dim]
        mean_err = dim_data["avg_procrustes_error"].mean()
        std_err = dim_data["avg_procrustes_error"].std()
        min_err = dim_data["avg_procrustes_error"].min()
        max_err = dim_data["avg_procrustes_error"].max()

        print(
            f"{dim:<6} {mean_err:<12.4f} {std_err:<12.4f} {min_err:<12.4f} {max_err:<12.4f}"
        )

    # Noise level analysis
    print("\nNoise level analysis:")
    print("Low noise (0.00-0.10):")
    low_noise = df[df["noise_level"] <= 0.10]
    print(f"  Mean error: {low_noise['avg_procrustes_error'].mean():.4f}")

    print("Medium noise (0.10-0.20):")
    med_noise = df[(df["noise_level"] > 0.10) & (df["noise_level"] <= 0.20)]
    print(f"  Mean error: {med_noise['avg_procrustes_error'].mean():.4f}")

    print("High noise (0.20-0.30):")
    high_noise = df[df["noise_level"] > 0.20]
    print(f"  Mean error: {high_noise['avg_procrustes_error'].mean():.4f}")

    # Failure analysis
    failed_runs = df[df["valid_runs"] < 10]
    if len(failed_runs) > 0:
        print("\nConfigurations with failures:")
        print(f"{'Dim':<6} {'Noise':<8} {'Valid Runs':<12} {'Avg Error':<12}")
        print("-" * 40)
        for _, row in failed_runs.iterrows():
            print(
                f"{row['dimension']:<6} {row['noise_level']:<8.2f} {row['valid_runs']:<12} {row['avg_procrustes_error']:<12.4f}"
            )


def main():
    """Run the noise robustness experiment."""
    np.random.seed(42)

    df = run_experiment()

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("Results saved to noise_robustness_results/")

    return df


if __name__ == "__main__":
    results_df = main()
