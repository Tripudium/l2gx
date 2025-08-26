"""
Plotting functions
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from l2gx.graphs import TGraph


def plot_patch_graph(
    patch_graph: TGraph,
    original_graph: TGraph = None,
    filename: str | None = None,
    show_legend: bool = False,
    show_title: bool = False,
):
    """
    Plot the original graph with partition coloring (left) and the patch graph (right) side by side.

    Args:
        patch_graph: The patch graph with patches and partition information.
        original_graph: The original graph to plot with partition coloring. If None, will be extracted from patch_graph.
        filename: The filename to save the plot to.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left plot: Original graph with nodes colored by partition
    if original_graph is None:
        raise ValueError(
            "original_graph must be provided to plot the original graph with partition coloring"
        )

    original_nx = original_graph.to_networkx()

    # Check if partition attribute exists
    if not hasattr(patch_graph, "partition"):
        raise ValueError("patch_graph must have 'partition' attribute")

    node_colors = patch_graph.partition
    print(
        f"Using partition attribute: {len(node_colors)} nodes, {len(np.unique(node_colors))} clusters"
    )

    # Verify partition size matches original graph
    if len(node_colors) != original_graph.num_nodes:
        raise ValueError(
            f"Partition size ({len(node_colors)}) doesn't match original graph nodes ({original_graph.num_nodes})"
        )

    n_patches = len(patch_graph.patches)

    # Use colorblind-friendly palette
    # Colors selected to be distinguishable for deuteranopia, protanopia, and tritanopia
    colorblind_friendly_colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
        "#aec7e8",  # Light Blue
        "#ffbb78",  # Light Orange
        "#98df8a",  # Light Green
        "#ff9896",  # Light Red
        "#c5b0d5",  # Light Purple
        "#c49c94",  # Light Brown
        "#f7b6d3",  # Light Pink
        "#c7c7c7",  # Light Gray
        "#dbdb8d",  # Light Olive
        "#9edae5",  # Light Cyan
    ]

    # Use colorblind-friendly colors, cycling if more patches than colors
    colors = [
        colorblind_friendly_colors[i % len(colorblind_friendly_colors)]
        for i in range(n_patches)
    ]

    # Create patch-based coloring instead of partition-based
    print("Creating patch-based node coloring...")

    # Map each node to the patches it belongs to
    node_to_patches = {}
    for patch_idx, patch in enumerate(patch_graph.patches):
        for node in patch.nodes:
            if node not in node_to_patches:
                node_to_patches[node] = []
            node_to_patches[node].append(patch_idx)

    # Create colors for each node based on patch membership
    node_color_list = []
    for node in range(original_graph.num_nodes):
        if node in node_to_patches:
            patch_indices = node_to_patches[node]
            if len(patch_indices) == 1:
                # Node belongs to only one patch
                node_color_list.append(colors[patch_indices[0]])
            else:
                # Node belongs to multiple patches - blend colors
                patch_color_strings = [colors[i] for i in patch_indices]
                # Convert hex colors to RGB values for blending
                patch_rgb = [mcolors.to_rgb(color) for color in patch_color_strings]
                # Average the RGB values
                blended_rgb = np.mean(patch_rgb, axis=0)
                node_color_list.append(blended_rgb)
        else:
            # Node doesn't belong to any patch (shouldn't happen normally)
            node_color_list.append("lightgray")

    print("Computing clustered layout for original graph...")
    # Create a layout that clusters nodes by partition
    pos_graph = {}

    # First get partition assignments for each node
    unique_partitions = np.unique(node_colors)
    n_partitions = len(unique_partitions)

    # Calculate cluster centers in a circular arrangement
    cluster_centers = {}
    for i, partition in enumerate(unique_partitions):
        angle = 2 * np.pi * i / n_partitions
        cluster_centers[partition] = np.array([np.cos(angle), np.sin(angle)]) * 3

    # For each partition, create a subgraph and layout nodes within cluster
    for partition in unique_partitions:
        partition_nodes = [
            node for node in range(len(node_colors)) if node_colors[node] == partition
        ]

        if len(partition_nodes) == 1:
            # Single node - place at cluster center
            pos_graph[partition_nodes[0]] = cluster_centers[partition]
        else:
            # Multiple nodes - create subgraph and layout around cluster center
            subgraph = original_nx.subgraph(partition_nodes)

            # Create more organic blob-like clusters
            if len(partition_nodes) <= 3:
                # For small clusters, use simple random positioning around center
                for j, node in enumerate(partition_nodes):
                    angle = np.random.uniform(0, 2 * np.pi)
                    radius = np.random.uniform(0.2, 0.8)
                    offset = np.array([np.cos(angle), np.sin(angle)]) * radius
                    pos_graph[node] = cluster_centers[partition] + offset
            else:
                # For larger clusters, use spring layout with random perturbations
                sub_pos = nx.spring_layout(
                    subgraph, k=0.1, iterations=100, seed=None
                )  # No fixed seed for randomness

                # Add random perturbations and scale for blob-like appearance
                for node, pos in sub_pos.items():
                    # Add random jitter for more organic appearance
                    jitter = np.random.normal(0, 0.15, 2)  # Random noise
                    # Scale and add cluster-specific randomness
                    scaled_pos = np.array(pos) * np.random.uniform(0.6, 1.0) + jitter
                    pos_graph[node] = cluster_centers[partition] + scaled_pos

    nx.draw_networkx_nodes(
        original_nx,
        pos_graph,
        node_color=node_color_list,
        node_size=30,  # Slightly larger nodes for better visibility
        alpha=0.7,
        ax=ax1,
    )
    # Sparsify edges for cleaner visualization - show only 10% of edges
    all_edges = list(original_nx.edges())
    if len(all_edges) > 0:
        # Randomly sample 10% of edges
        np.random.seed(42)  # For reproducible sparsification
        n_edges_to_show = max(1, int(len(all_edges) * 0.1))
        sampled_edges = np.random.choice(
            len(all_edges), size=n_edges_to_show, replace=False
        )
        edges_to_draw = [all_edges[i] for i in sampled_edges]

        # Create a subgraph with only the sampled edges
        sparse_graph = nx.Graph()
        sparse_graph.add_nodes_from(original_nx.nodes())
        sparse_graph.add_edges_from(edges_to_draw)

        nx.draw_networkx_edges(
            sparse_graph,
            pos_graph,
            alpha=0.4,  # Increased from 0.1 to make edges more visible
            width=0.8,  # Increased from 0.3 to make edges thicker
            edge_color="darkgray",  # Darker color for better visibility
            ax=ax1,
        )
    # Count overlapping nodes for title
    overlapping_nodes = sum(1 for node in node_to_patches.values() if len(node) > 1)

    if show_title:
        ax1.set_title(
            f"Graph\n({original_graph.num_nodes} nodes, {n_patches} patches, {overlapping_nodes} overlapping)",
            fontsize=14,
            fontweight="bold",
        )
    ax1.axis("off")

    if show_legend:
        legend_elements = [
            plt.scatter([], [], c=colors[i], s=100, alpha=0.8, label=f"Patch {i + 1}")
            for i in range(min(n_patches, 10))
        ]  # Limit legend to 10 patches
        if n_patches > 10:
            legend_elements.append(
                plt.scatter([], [], c="white", s=100, alpha=0.8, label="...")
            )
        ax1.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 1))

    # Right plot: Patch graph
    patch_nx = patch_graph.to_networkx()

    # Calculate node sizes based on patch sizes
    patch_sizes = [len(patch.nodes) for patch in patch_graph.patches]
    max_size = max(patch_sizes)
    min_size = min(patch_sizes)
    # Scale node sizes between 500 and 1500 (increased from 200-1000)
    if max_size == min_size:
        node_sizes = [1200] * len(patch_sizes)  # All patches same size
    else:
        node_sizes = [
            500 + 1200 * (size - min_size) / (max_size - min_size)
            for size in patch_sizes
        ]

    # Use circular layout for patch graph
    pos_patch = nx.circular_layout(patch_nx)

    nx.draw_networkx_nodes(
        patch_nx,
        pos_patch,
        node_color=colors[: len(patch_graph.patches)],
        node_size=node_sizes,
        alpha=1.0,  # Make circles solid (fully opaque)
        ax=ax2,
    )

    # Draw patch graph edges with thickness based on overlap size
    if hasattr(patch_graph, "overlap_nodes"):
        edge_weights = []
        for edge in patch_nx.edges():
            i, j = edge
            overlap_key = (i, j) if (i, j) in patch_graph.overlap_nodes else (j, i)
            if overlap_key in patch_graph.overlap_nodes:
                overlap_size = len(patch_graph.overlap_nodes[overlap_key])
                edge_weights.append(overlap_size)
            else:
                edge_weights.append(1)

        # Normalize edge weights for visualization
        if edge_weights:
            max_weight = max(edge_weights)
            edge_widths = [1 + 4 * w / max_weight for w in edge_weights]
        else:
            edge_widths = [1] * len(patch_nx.edges())

        nx.draw_networkx_edges(
            patch_nx,
            pos_patch,
            width=edge_widths,
            alpha=0.6,
            edge_color="darkblue",
            arrows=False,  # Remove arrows from edges
            ax=ax2,
        )
    else:
        nx.draw_networkx_edges(
            patch_nx,
            pos_patch,
            width=2,
            alpha=0.6,
            edge_color="darkblue",
            arrows=False,  # Remove arrows from edges
            ax=ax2,
        )

    # Add patch labels (numbered 1 to N, without patch sizes)
    nx.draw_networkx_labels(
        patch_nx,
        pos_patch,
        {i: str(i + 1) for i in range(len(patch_graph.patches))},
        font_size=12,
        font_weight="bold",
        font_color="white",  # White text for better contrast on colored nodes
        ax=ax2,
    )

    if show_title:
        ax2.set_title(
            f"Patch Graph\n({len(patch_graph.patches)} patches, {patch_nx.number_of_edges()} connections)",
            fontsize=14,
            fontweight="bold",
        )
    ax2.axis("off")

    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

    return fig
