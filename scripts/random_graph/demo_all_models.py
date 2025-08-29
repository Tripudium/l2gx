#!/usr/bin/env python3
"""
Comprehensive demo of all random graph models for patch generation testing.

This script demonstrates and compares:
1. Community-Affiliation Graph Model (AGM)
2. Overlapping Stochastic Block Model (OSBM)  
3. Petti-Vempala Random Overlapping Communities (ROC) Model
4. General Overlapping Communities Model

Each model has different strengths for testing patch-based algorithms:
- AGM: Good for testing dense, well-defined communities
- OSBM: Good for testing block-structured overlapping communities
- Petti-Vempala ROC: Good for theoretical analysis with controlled overlap
- General Overlapping: Good for realistic, grown communities with varying strengths
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from agm import CommunityAffiliationGraphModel
from osbm import OverlappingStochasticBlockModel
from petti_vempala_roc import PettiVempalaROC
from general_overlapping_communities import GeneralOverlappingCommunities


def compare_all_models():
    """Compare all four models with similar parameter settings."""
    print("COMPARATIVE ANALYSIS OF RANDOM GRAPH MODELS")
    print("=" * 60)

    # Common parameters
    common_params = {"n_nodes": 120, "random_state": 42}

    # Model-specific configurations
    models_config = [
        {
            "name": "AGM",
            "class": CommunityAffiliationGraphModel,
            "params": {
                **common_params,
                "n_communities": 5,
                "community_size_dist": "uniform",
                "community_size_params": {"min_size": 15, "max_size": 30},
                "intra_community_prob": 0.3,
                "background_prob": 0.01,
            },
        },
        {
            "name": "OSBM",
            "class": OverlappingStochasticBlockModel,
            "params": {
                **common_params,
                "n_communities": 5,
                "community_assignments": "balanced",
                "overlap_probability": 0.25,
                "intra_community_prob": 0.3,
                "inter_community_prob": 0.02,
            },
        },
        {
            "name": "Petti-Vempala ROC",
            "class": PettiVempalaROC,
            "params": {
                **common_params,
                "k_communities": 5,
                "p_in": 0.3,
                "p_out": 0.02,
                "overlap_fraction": 0.2,
            },
        },
        {
            "name": "General Overlapping",
            "class": GeneralOverlappingCommunities,
            "params": {
                **common_params,
                "n_communities": 5,
                "community_generation": "grown",
                "average_community_size": 24,
                "overlap_factor": 0.25,
                "membership_strength_dist": "uniform",
                "base_edge_prob": 0.15,
            },
        },
    ]

    results = []
    graphs = {}

    # Generate graphs and collect statistics
    for config in models_config:
        print(f"\nGenerating {config['name']} model...")

        model = config["class"](**config["params"])
        G = model.generate_graph()
        graphs[config["name"]] = (model, G)

        stats = model.compute_statistics()
        stats["model"] = config["name"]
        results.append(stats)

        # Print key statistics
        print(f"  Nodes: {stats['n_nodes']}, Edges: {stats['n_edges']}")
        print(f"  Density: {stats['density']:.4f}")
        print(f"  Avg communities per node: {stats['avg_communities_per_node']:.2f}")
        print(
            f"  Overlap ratio: {stats.get('overlap_ratio', stats['nodes_in_multiple_communities'] / stats['n_nodes']):.2f}"
        )
        print(f"  Clustering: {stats['clustering_coefficient']:.3f}")

    # Create comparison table
    df = pd.DataFrame(results)

    print(f"\n{'=' * 60}")
    print("COMPARISON TABLE")
    print(f"{'=' * 60}")

    comparison_cols = [
        "model",
        "n_edges",
        "density",
        "avg_communities_per_node",
        "nodes_in_multiple_communities",
        "clustering_coefficient",
    ]

    print(df[comparison_cols].to_string(index=False, float_format="%.3f"))

    return graphs, df


def visualize_comparison(graphs):
    """Create visualization comparing all three models."""
    print(f"\nCreating comparison visualizations...")

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for i, (model_name, (model, graph)) in enumerate(graphs.items()):
        # Get layout for consistency
        pos = graph.nodes(data=False)
        if hasattr(model, "graph"):
            import networkx as nx

            pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)

        # Plot 1: Basic graph structure
        node_colors = [
            len(model.node_communities.get(node, {})) for node in graph.nodes()
        ]
        im1 = axes[i, 0].scatter(
            [pos[node][0] for node in graph.nodes()],
            [pos[node][1] for node in graph.nodes()],
            c=node_colors,
            cmap="viridis",
            s=30,
        )

        # Draw edges
        for edge in graph.edges():
            x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
            y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
            axes[i, 0].plot(x_coords, y_coords, "k-", alpha=0.1, linewidth=0.5)

        axes[i, 0].set_title(f"{model_name}: Nodes by # communities")
        axes[i, 0].set_aspect("equal")

        # Plot 2: Community size distribution
        if hasattr(model, "community_nodes"):
            community_sizes = [len(nodes) for nodes in model.community_nodes.values()]
            axes[i, 1].hist(community_sizes, bins=10, alpha=0.7, edgecolor="black")
            axes[i, 1].set_title(f"{model_name}: Community sizes")
            axes[i, 1].set_xlabel("Community size")
            axes[i, 1].set_ylabel("Count")

        # Plot 3: Node membership distribution
        if hasattr(model, "node_communities"):
            node_memberships = [len(comms) for comms in model.node_communities.values()]
            axes[i, 2].hist(
                node_memberships,
                bins=range(1, max(node_memberships) + 2),
                alpha=0.7,
                edgecolor="black",
                align="left",
            )
            axes[i, 2].set_title(f"{model_name}: Memberships per node")
            axes[i, 2].set_xlabel("# communities per node")
            axes[i, 2].set_ylabel("Count")

    plt.tight_layout()

    # Save visualization
    output_dir = Path("model_comparison_plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Visualization saved to {output_dir / 'model_comparison.png'}")
    plt.show()


def analyze_patch_suitability(graphs):
    """Analyze how suitable each model is for patch-based algorithm testing."""
    print(f"\n{'=' * 60}")
    print("PATCH-BASED ALGORITHM SUITABILITY ANALYSIS")
    print(f"{'=' * 60}")

    suitability_analysis = {
        "AGM": {
            "strengths": [
                "Well-defined, dense communities",
                "Controllable overlap structure",
                "Good for testing basic patch detection",
                "Fast generation",
            ],
            "weaknesses": [
                "Communities may be too artificial",
                "Limited growth patterns",
                "Binary membership only",
            ],
            "best_for": "Testing patch detection on clear community structure",
        },
        "OSBM": {
            "strengths": [
                "Principled probabilistic model",
                "Flexible mixing parameters",
                "Good theoretical foundation",
                "Handles complex overlap patterns",
            ],
            "weaknesses": [
                "May create overly regular structures",
                "Limited community size variation",
                "Complex parameter tuning",
            ],
            "best_for": "Testing alignment algorithms on block-structured data",
        },
        "ROC": {
            "strengths": [
                "Realistic community growth patterns",
                "Variable membership strengths",
                "Flexible generation mechanisms",
                "Good for hierarchical structures",
            ],
            "weaknesses": [
                "More complex parameter space",
                "Slower generation",
                "Less theoretical guarantees",
            ],
            "best_for": "Testing on realistic social network-like structures",
        },
    }

    for model_name, analysis in suitability_analysis.items():
        print(f"\n{model_name} Model:")
        print(f"  Strengths:")
        for strength in analysis["strengths"]:
            print(f"    + {strength}")
        print(f"  Weaknesses:")
        for weakness in analysis["weaknesses"]:
            print(f"    - {weakness}")
        print(f"  Best for: {analysis['best_for']}")


def generate_test_cases():
    """Generate specific test cases suitable for patch-based algorithm testing."""
    print(f"\n{'=' * 60}")
    print("GENERATING SPECIFIC TEST CASES")
    print(f"{'=' * 60}")

    test_cases = [
        {
            "name": "Dense Overlapping Patches",
            "model": "AGM",
            "params": {
                "n_nodes": 200,
                "n_communities": 8,
                "community_size_dist": "uniform",
                "community_size_params": {"min_size": 20, "max_size": 40},
                "intra_community_prob": 0.4,
                "background_prob": 0.005,
            },
            "use_case": "Testing patch alignment with high intra-patch connectivity",
        },
        {
            "name": "Sparse Block Structure",
            "model": "OSBM",
            "params": {
                "n_nodes": 150,
                "n_communities": 6,
                "community_assignments": "power_law",
                "overlap_probability": 0.15,
                "intra_community_prob": 0.25,
                "inter_community_prob": 0.01,
            },
            "use_case": "Testing on sparse graphs with clear block structure",
        },
        {
            "name": "Realistic Social Structure",
            "model": "ROC",
            "params": {
                "n_nodes": 300,
                "n_communities": 12,
                "community_generation": "preferential",
                "average_community_size": 30,
                "overlap_factor": 0.3,
                "membership_strength_dist": "beta",
                "edge_prob_function": "strength_product",
            },
            "use_case": "Testing on realistic community structures with growth patterns",
        },
    ]

    # Generate and save test cases
    output_dir = Path("test_cases")
    output_dir.mkdir(exist_ok=True)

    for case in test_cases:
        print(f"\nGenerating: {case['name']}")

        if case["model"] == "AGM":
            model = CommunityAffiliationGraphModel(random_state=42, **case["params"])
        elif case["model"] == "OSBM":
            model = OverlappingStochasticBlockModel(random_state=42, **case["params"])
        elif case["model"] == "ROC":
            model = RandomOverlappingCommunities(random_state=42, **case["params"])

        G = model.generate_graph()
        stats = model.compute_statistics()

        print(f"  Generated: {stats['n_nodes']} nodes, {stats['n_edges']} edges")
        print(f"  Use case: {case['use_case']}")

        # Save graph and community information
        import networkx as nx

        graph_file = output_dir / f"{case['name'].lower().replace(' ', '_')}.graphml"
        nx.write_graphml(G, graph_file)
        print(f"  Saved to: {graph_file}")


def main():
    """Run complete demonstration and analysis."""
    print("RANDOM GRAPH MODELS FOR PATCH GENERATION TESTING")
    print("=" * 60)
    print("This demo showcases three models for generating test graphs:")
    print("1. Community-Affiliation Graph Model (AGM)")
    print("2. Overlapping Stochastic Block Model (OSBM)")
    print("3. Random Overlapping Communities (ROC) Model")
    print()

    # Compare all models
    graphs, comparison_df = compare_all_models()

    # Create visualizations
    visualize_comparison(graphs)

    # Analyze suitability for patch-based algorithms
    analyze_patch_suitability(graphs)

    # Generate specific test cases
    generate_test_cases()

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print("✓ Implemented three overlapping community models")
    print("✓ Generated comparison statistics and visualizations")
    print("✓ Analyzed suitability for patch-based algorithm testing")
    print("✓ Created specific test cases for different scenarios")
    print()
    print("Next steps:")
    print("- Use these models to generate test graphs for patch alignment")
    print("- Compare patch detection performance across different structures")
    print("- Test alignment algorithms under various overlap conditions")
    print("- Validate embedding quality on known community structures")


if __name__ == "__main__":
    main()
