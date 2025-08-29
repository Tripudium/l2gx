#!/usr/bin/env python3
"""
Create PDF/PNG Embedding Visualizations

Creates enhanced embedding visualizations from any embedding file (.npz, .npy, .pt)
using datashader for better visibility.

Usage:
    python create_pdf.py embedding_file.npz                    # Unlabeled plot
    python create_pdf.py embedding_file.npz --labels labels    # With labels from npz
    python create_pdf.py embedding.npy --labels labels.npy     # Separate label file
    python create_pdf.py embedding.npz --output my_plot        # Custom output name
    python create_pdf.py embedding.npz --no-enhance            # Without enhancement
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple

import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import umap
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_embedding(filename: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load embedding and optionally labels from file.
    
    Args:
        filename: Path to embedding file (.npz, .npy, .pt)
        
    Returns:
        Tuple of (embedding, labels or None)
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    
    if path.suffix == '.npz':
        # Load from npz file
        data = np.load(filename, allow_pickle=True)
        
        # Try different common keys for embeddings
        embedding = None
        for key in ['embedding', 'embeddings', 'X', 'x', 'features', 'data']:
            if key in data.files:
                embedding = data[key]
                break
        
        if embedding is None:
            # If no standard key found, try the first array-like item
            if len(data.files) > 0:
                embedding = data[data.files[0]]
            else:
                raise ValueError(f"No embedding found in {filename}")
        
        # Try to find labels
        labels = None
        for key in ['labels', 'y', 'Y', 'targets', 'classes']:
            if key in data.files:
                labels = data[key]
                break
                
        return embedding, labels
        
    elif path.suffix == '.npy':
        # Load from npy file
        embedding = np.load(filename)
        return embedding, None
        
    elif path.suffix == '.pt' or path.suffix == '.pth':
        # Load from PyTorch file
        data = torch.load(filename, map_location='cpu')
        
        if isinstance(data, torch.Tensor):
            return data.numpy(), None
        elif isinstance(data, dict):
            # Try to find embedding and labels
            embedding = None
            for key in ['embedding', 'embeddings', 'X', 'x', 'features']:
                if key in data:
                    embedding = data[key]
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.numpy()
                    break
            
            labels = None
            for key in ['labels', 'y', 'Y', 'targets']:
                if key in data:
                    labels = data[key]
                    if isinstance(labels, torch.Tensor):
                        labels = labels.numpy()
                    break
                    
            if embedding is None:
                raise ValueError(f"No embedding found in {filename}")
            return embedding, labels
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def load_labels(filename: str) -> np.ndarray:
    """Load labels from a separate file."""
    path = Path(filename)
    
    if path.suffix == '.npz':
        data = np.load(filename, allow_pickle=True)
        # Try different keys
        for key in ['labels', 'y', 'Y', 'targets', 'classes']:
            if key in data.files:
                return data[key]
        # Default to first array
        return data[data.files[0]]
        
    elif path.suffix == '.npy':
        return np.load(filename)
        
    elif path.suffix in ['.pt', '.pth']:
        data = torch.load(filename, map_location='cpu')
        if isinstance(data, torch.Tensor):
            return data.numpy()
        else:
            return data
    else:
        raise ValueError(f"Unsupported label file format: {path.suffix}")


def create_enhanced_points_df(embedding: np.ndarray, labels: Optional[np.ndarray] = None):
    """Create DataFrame with enhanced points for better visibility."""
    
    # Create UMAP projection
    print("Creating UMAP projection...")
    umap_coords = umap.UMAP(
        n_neighbors=5,  # Fewer neighbors for more spread
        min_dist=0.5,   # Larger minimum distance for more spread
        spread=2.0,     # Increase spread parameter
        random_state=42,
    ).fit_transform(embedding)
    
    # If no labels provided, use zeros
    if labels is None:
        labels = np.zeros(len(embedding), dtype=int)
    
    # Create multiple points around each original point for thickness
    expanded_data = []
    
    # Offset points for enhanced visibility
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
    
    for i in range(len(umap_coords)):
        x, y = umap_coords[i]
        label = labels[i]
        
        # Add the original point
        expanded_data.append([x, y, label])
        
        # Add nearby points for enhanced visibility
        for dx, dy in offsets:
            expanded_data.append([x + dx, y + dy, label])
    
    df = pl.DataFrame(expanded_data, schema=["x", "y", "class"], orient="row")
    # Convert class to string then categorical for polars
    df = df.with_columns(pl.col("class").cast(pl.Utf8).cast(pl.Categorical))
    
    return df, umap_coords


def create_visualization(
    embedding: np.ndarray,
    labels: Optional[np.ndarray] = None,
    output_name: str = "embedding_visualization",
    enhance: bool = True,
    canvas_size: int = 400,
    dpi: int = 300,
    figsize: Tuple[float, float] = (8, 8),
) -> None:
    """Create PDF and PNG visualizations of embeddings.
    
    Args:
        embedding: The embedding matrix
        labels: Optional class labels
        output_name: Base name for output files (without extension)
        enhance: Whether to enhance point visibility
        canvas_size: Size of datashader canvas
        dpi: DPI for saved figures
        figsize: Figure size in inches
    """
    
    print(f"Processing embedding with shape {embedding.shape}")
    
    # Create enhanced or simple dataframe
    if enhance:
        print("Creating enhanced point dataset...")
        df, umap_coords = create_enhanced_points_df(embedding, labels)
    else:
        print("Creating UMAP projection...")
        umap_coords = umap.UMAP(
            n_neighbors=5,
            min_dist=0.5,
            spread=2.0,
            random_state=42,
        ).fit_transform(embedding)
        
        if labels is None:
            labels = np.zeros(len(embedding), dtype=int)
            
        df = pl.DataFrame({
            "x": umap_coords[:, 0],
            "y": umap_coords[:, 1],
            "class": labels.astype(int),
        }).with_columns(pl.col("class").cast(pl.Utf8).cast(pl.Categorical))
    
    print("Creating datashader visualization...")
    
    # Create canvas
    canvas = ds.Canvas(
        plot_width=canvas_size,
        plot_height=canvas_size,
        x_range=(df["x"].min(), df["x"].max()),
        y_range=(df["y"].min(), df["y"].max()),
    )
    
    # Aggregate points by class
    agg = canvas.points(df.to_pandas(), "x", "y", ds.count_cat("class"))
    
    # Determine number of unique classes
    n_classes = len(df["class"].unique())
    
    # Use bright, vibrant colors
    if n_classes <= 1:
        # Single class - use blue
        colors = ["#0080FF"]
    elif n_classes <= 7:
        # Few classes - use distinct colors
        colors = [
            "#0080FF",  # Blue
            "#FF8000",  # Orange
            "#00C000",  # Green
            "#FF4040",  # Red
            "#8040FF",  # Purple
            "#C0C000",  # Yellow
            "#FF4080",  # Pink
        ]
    else:
        # Many classes - generate colors
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20')
        colors = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" 
                  for r, g, b, _ in [cmap(i/n_classes) for i in range(n_classes)]]
    
    # Create image with log shading
    img = tf.shade(
        agg,
        color_key={str(i): colors[i % len(colors)] for i in range(n_classes)},
        how="log",
        alpha=255,
    )
    
    # Set white background
    img = tf.set_background(img, "white")
    
    # Convert to array
    img_array = np.array(img.to_pil())
    
    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Display the image
    ax.imshow(img_array, aspect="equal", interpolation="nearest")
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove all margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    # Save as PDF
    pdf_file = f"{output_name}.pdf"
    print(f"Saving PDF: {pdf_file}")
    with PdfPages(pdf_file) as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0, dpi=dpi)
    
    # Save as PNG
    png_file = f"{output_name}.png"
    print(f"Saving PNG: {png_file}")
    fig.savefig(png_file, bbox_inches="tight", pad_inches=0, dpi=dpi)
    
    plt.close(fig)
    
    # Print summary
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETED")
    print("=" * 60)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Number of points: {len(embedding)}")
    if labels is not None and not np.all(labels == 0):
        print(f"Number of classes: {len(np.unique(labels))}")
    else:
        print("No class labels (unlabeled visualization)")
    print(f"Enhancement: {'Enabled' if enhance else 'Disabled'}")
    if enhance:
        print(f"Points expanded: {len(df)} total ({len(df) // 13} original)")
    print(f"\nFiles created:")
    print(f"  - {pdf_file}")
    print(f"  - {png_file}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Create enhanced embedding visualizations using datashader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_pdf.py embedding.npz                      # Unlabeled plot
  python create_pdf.py embedding.npz --labels labels      # Labels from npz key
  python create_pdf.py embedding.npy --labels labels.npy  # Separate label file
  python create_pdf.py embedding.npz --output my_plot     # Custom output name
  python create_pdf.py embedding.npz --no-enhance         # Without enhancement
  python create_pdf.py embedding.npz --canvas-size 600    # Larger canvas
        """
    )
    
    parser.add_argument(
        "embedding_file",
        help="Path to embedding file (.npz, .npy, .pt)"
    )
    parser.add_argument(
        "--labels",
        help="Labels: either a key name (for npz) or path to label file",
        default=None
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file name (without extension)",
        default="embedding_visualization"
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disable point enhancement for visibility"
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=400,
        help="Datashader canvas size (default: 400)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures (default: 300)"
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[8, 8],
        help="Figure size in inches (default: 8 8)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load embedding
        embedding, embedded_labels = load_embedding(args.embedding_file)
        
        # Handle labels
        labels = None
        if args.labels:
            if embedded_labels is not None and args.labels in ['labels', 'y', 'Y']:
                # Use embedded labels
                labels = embedded_labels
                print(f"Using labels from embedding file")
            elif Path(args.labels).exists():
                # Load from separate file
                labels = load_labels(args.labels)
                print(f"Loaded labels from {args.labels}")
            else:
                print(f"Warning: Could not find labels '{args.labels}', creating unlabeled plot")
        elif embedded_labels is not None:
            # Use embedded labels if available and no label argument given
            labels = embedded_labels
            print("Using labels found in embedding file")
        
        # Create visualization
        create_visualization(
            embedding,
            labels,
            output_name=args.output,
            enhance=not args.no_enhance,
            canvas_size=args.canvas_size,
            dpi=args.dpi,
            figsize=tuple(args.figsize)
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())