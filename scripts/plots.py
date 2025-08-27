"""
Plotting functions for L2GX.

This module provides plotting utilities for:
- Patch graph visualization
- UMAP embeddings with datashader
"""

import warnings
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from l2gx.graphs import TGraph

# Optional imports for UMAP plotting
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import datashader as ds
    import datashader.transfer_functions as tf
    DATASHADER_AVAILABLE = True
except ImportError:
    DATASHADER_AVAILABLE = False

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    try:
        import pandas as pd
        PANDAS_AVAILABLE = True
    except ImportError:
        PANDAS_AVAILABLE = False


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


def plot_datashader_umap(
    embedding: np.ndarray,
    labels: np.ndarray | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    umap_params: dict | None = None,
    datashader_params: dict | None = None,
    colors: list[str] | None = None,
    figsize: tuple[float, float] = (8, 8),
    dpi: int = 150,
    show_plot: bool = True,
    verbose: bool = True,
    enhance_visibility: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Create a high-quality UMAP visualization using datashader.

    This function creates publication-quality UMAP plots similar to l2g_embeddings.pdf,
    using datashader for better handling of overlapping points and large datasets.

    Parameters
    ----------
    embedding : np.ndarray
        The embedding matrix of shape (n_samples, n_features).
    labels : np.ndarray, optional
        Class labels for each sample. If None, all points are treated as one class.
    title : str, optional
        Title for the plot.
    save_path : str or Path, optional
        Path to save the plot (supports .pdf, .png, .jpg).
    umap_params : dict, optional
        Parameters for UMAP. Default: {'n_neighbors': 15, 'min_dist': 0.1, 'random_state': 42}
    datashader_params : dict, optional
        Parameters for datashader. Default: {'width': 600, 'height': 600}
    colors : list of str, optional
        List of hex colors for classes. If None, uses default bright colors.
    figsize : tuple, optional
        Figure size in inches. Default: (8, 8)
    dpi : int, optional
        DPI for the saved figure. Default: 150
    show_plot : bool, optional
        Whether to display the plot. Default: True
    verbose : bool, optional
        Whether to print progress messages. Default: True
    enhance_visibility : bool, optional
        Whether to enhance point visibility by adding multiple points per sample. Default: True

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    umap_coords : np.ndarray
        The 2D UMAP coordinates of shape (n_samples, 2).

    Examples
    --------
    >>> from l2gx.embedding import get_embedding
    >>> from l2gx.datasets import get_dataset
    >>> from scripts.plots import plot_datashader_umap
    >>>
    >>> # Get embedding
    >>> dataset = get_dataset("Cora")
    >>> embedder = get_embedding("vgae", embedding_dim=64)
    >>> embedding = embedder.fit_transform(dataset.to("torch-geometric"))
    >>>
    >>> # Create plot
    >>> fig, coords = plot_datashader_umap(
    ...     embedding,
    ...     labels=dataset.y,
    ...     title="Cora VGAE Embeddings",
    ...     save_path="cora_umap.pdf"
    ... )
    """

    # Check requirements
    if not UMAP_AVAILABLE:
        raise ImportError(
            "UMAP is not available. Please install it with: pip install umap-learn"
        )
    if not DATASHADER_AVAILABLE:
        raise ImportError(
            "Datashader is not available. Please install it with: pip install datashader"
        )
    if not POLARS_AVAILABLE and not PANDAS_AVAILABLE:
        raise ImportError(
            "Neither polars nor pandas is available. Please install one: pip install polars"
        )

    if verbose:
        print(f"Creating UMAP visualization for embedding of shape {embedding.shape}")

    # Handle labels
    if labels is None:
        labels = np.zeros(len(embedding), dtype=int)
    else:
        labels = np.asarray(labels)

    # Set default parameters to match l2g_embeddings.pdf exactly
    if umap_params is None:
        umap_params = {
            'n_neighbors': 5,    # Fewer neighbors for more spread (matches original)
            'min_dist': 0.5,     # Larger minimum distance for more spread (matches original)
            'spread': 2.0,       # Increase spread parameter (matches original)
            'random_state': 42,
            'n_components': 2,
        }

    if datashader_params is None:
        datashader_params = {
            'width': 400,        # Smaller canvas for enhanced visibility (matches original)
            'height': 400,       # Smaller canvas for enhanced visibility (matches original)
        }

    # Default colors (bright and vibrant)
    if colors is None:
        colors = [
            "#0080FF",  # Blue
            "#FF8000",  # Orange
            "#00C000",  # Green
            "#FF4040",  # Red
            "#8040FF",  # Purple
            "#C0C000",  # Yellow
            "#FF4080",  # Pink
            "#00C0C0",  # Cyan
            "#804040",  # Brown
            "#408040",  # Dark green
        ]

    # Step 1: Compute UMAP embedding
    if verbose:
        print("Computing UMAP embedding...")

    reducer = umap.UMAP(**umap_params)
    umap_coords = reducer.fit_transform(embedding)

    if verbose:
        print(f"UMAP complete. Coordinates shape: {umap_coords.shape}")

    # Step 2: Create DataFrame for datashader
    if enhance_visibility:
        # Create multiple points per sample for enhanced visibility
        df = _create_enhanced_dataframe(umap_coords, labels)
    else:
        # Simple DataFrame
        df = _create_simple_dataframe(umap_coords, labels)

    # Step 3: Create datashader image
    if verbose:
        print("Creating datashader visualization...")

    # Create canvas
    canvas = ds.Canvas(
        plot_width=datashader_params['width'],
        plot_height=datashader_params['height'],
        x_range=(df['x'].min(), df['x'].max()),
        y_range=(df['y'].min(), df['y'].max()),
    )

    # Convert to pandas for datashader if using polars
    if POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
        df_pandas = df.to_pandas()
    else:
        df_pandas = df

    # Aggregate points by class
    agg = canvas.points(df_pandas, 'x', 'y', ds.count_cat('class'))

    # Create color mapping
    n_classes = len(np.unique(labels))
    color_map = {str(i): colors[i % len(colors)] for i in range(n_classes)}

    # Shade the image
    img = tf.shade(
        agg,
        color_key=color_map,
        how='log',  # Log scale for better visibility
        alpha=255,  # Full opacity
    )

    # Set white background
    img = tf.set_background(img, 'white')

    # Step 4: Create matplotlib figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Convert datashader image to numpy array
    img_array = np.array(img.to_pil())

    # Display the image
    ax.imshow(img_array, aspect='equal', interpolation='nearest')
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove all margins and padding for clean plot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Step 5: Save if requested
    if save_path:
        save_path = Path(save_path)
        if verbose:
            print(f"Saving plot to {save_path}")

        if save_path.suffix == '.pdf':
            # Save as PDF without any padding or text
            with PdfPages(save_path) as pdf:
                pdf.savefig(fig, bbox_inches='tight', pad_inches=0, dpi=300)
        else:
            # Save as image without any padding 
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)

    # Step 6: Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, umap_coords


def _create_enhanced_dataframe(coords: np.ndarray, labels: np.ndarray):
    """Create DataFrame with enhanced points for better visibility."""

    expanded_data = []

    # Offset points matching create_pdf.py exactly for thick, visible points
    offsets = [
        (0.02, 0),
        (-0.02, 0),
        (0, 0.02),
        (0, -0.02),
        (0.01, 0.01),
        (-0.01, 0.01),
        (0.01, -0.01),
        (-0.01, -0.01),
        (0.015, 0),
        (-0.015, 0),
        (0, 0.015),
        (0, -0.015),
    ]

    for i in range(len(coords)):
        x, y = coords[i]
        label = labels[i]

        # Add the original point
        expanded_data.append([x, y, str(label)])

        # Add nearby points for enhanced visibility (matches create_pdf.py)
        for dx, dy in offsets:
            expanded_data.append([
                x + dx,
                y + dy,
                str(label)  # Convert to string for categorical
            ])

    # Create DataFrame (prefer polars for performance)
    if POLARS_AVAILABLE:
        df = pl.DataFrame(
            expanded_data,
            schema=["x", "y", "class"],
            orient="row"
        )
        # Convert class to categorical
        df = df.with_columns(pl.col("class").cast(pl.Categorical))
    else:
        df = pd.DataFrame(expanded_data, columns=["x", "y", "class"])
        df["class"] = df["class"].astype('category')

    return df


def _create_simple_dataframe(coords: np.ndarray, labels: np.ndarray):
    """Create simple DataFrame without enhancement."""

    data = {
        'x': coords[:, 0],
        'y': coords[:, 1],
        'class': labels.astype(str),
    }

    if POLARS_AVAILABLE:
        df = pl.DataFrame(data)
        df = df.with_columns(pl.col("class").cast(pl.Categorical))
    else:
        df = pd.DataFrame(data)
        df["class"] = df["class"].astype('category')

    return df
