#!/usr/bin/env python3
"""
Plot Embeddings - Visualization tool for graph embeddings

This script uses the EmbeddingExperiment class to compute embeddings and then
creates comprehensive visualizations including:
- Main embedding UMAP plot
- Individual patch plots (if applicable)
- Patch grid overview (if applicable)

Usage: python run/plot_embeddings.py <config_file>

Example:
    python run/plot_embeddings.py configs/cora_patched.yaml
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Optional
from functools import partial

# Import core embedding functionality
from embedding_experiment import EmbeddingExperiment

# Plotting imports
import umap
import datashader as ds
import datashader.transfer_functions as tf
from datashader.mpl_ext import dsshow
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class EmbeddingVisualizer:
    """Handles visualization of embeddings computed by EmbeddingExperiment"""
    
    def __init__(self, experiment: EmbeddingExperiment):
        """Initialize with completed embedding experiment"""
        self.experiment = experiment
        self.embedding = experiment.embedding
        self.patches = experiment.patches
        self.data = experiment.data
        self.output_dir = experiment.output_dir
        
        # Visualization parameters (hardcoded defaults)
        self.umap_params = {
            'min_dist': 0.3,  # Increased from 0.0 for more spread out clusters
            'metric': 'euclidean'
        }
        self.plot_params = {
            'pointsize': 5,
            'size': 2.0,
            'dpi': 1200,
            'individual_patch_dpi': 600,
            'grid_dpi': 300
        }
        
    def plot_embedding_datashader(self, embedding: np.ndarray, labels: torch.Tensor, 
                                 output_file: Path, title: str = "") -> None:
        """Create datashader-style UMAP plot"""
        print(f"Creating UMAP plot: {output_file.name}...")
        
        # Create figure
        size = self.plot_params['size']
        dpi = self.plot_params['dpi']
        pointsize = self.plot_params['pointsize']
        
        fig = plt.figure(figsize=(size, size), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax_size = size * dpi
        pad = 2 * pointsize / ax_size
        
        # Convert labels to numpy
        y = labels.cpu().numpy() if hasattr(labels, 'cpu') else labels
        nodes = np.flatnonzero(y >= 0)  # Only labeled nodes
        
        # Apply UMAP transformation
        coords = embedding[nodes] if len(nodes) < len(embedding) else embedding
        vc = umap.UMAP(min_dist=self.umap_params['min_dist'], 
                      metric=self.umap_params['metric'], 
                      verbose=False).fit_transform(coords)
        
        # Calculate ranges with padding
        min_range = vc.min(axis=0)
        max_range = vc.max(axis=0)
        pad_vals = (max_range - min_range) * pad
        
        x_range = (min_range[0] - pad_vals[0], max_range[0] + pad_vals[0])
        y_range = (min_range[1] - pad_vals[1], max_range[1] + pad_vals[1])
        
        # Create DataFrame for datashader
        df = pd.DataFrame(vc, columns=['x', 'y'])
        df['label'] = y[nodes]
        df['label'] = df['label'].astype('category')
        
        # Generate brighter colors for better visibility
        num_labels = len(np.unique(y[y >= 0]))
        colors = sns.color_palette('husl', num_labels)
        # Make colors brighter by increasing saturation and ensuring minimum brightness
        colors = {i: tuple(min(255, int(vi * 255 * 1.3)) for vi in v) for i, v in enumerate(colors)}
        
        dsshow(df, ds.Point('x', 'y'), ds.count_cat('label'), ax=ax, norm='eq_hist', color_key=colors,
               shade_hook=partial(tf.dynspread, threshold=0.99, max_px=pointsize, shape='circle'), 
               alpha_range=(55, 255), x_range=x_range, y_range=y_range)
        
        ax.set_axis_off()
        plt.margins(0.01, 0.01)
        if title:
            plt.title(title, pad=20, fontsize=12)
        plt.savefig(output_file, dpi=dpi)
        plt.close()
        
        print(f" Plot saved: {output_file}")
    
    def create_main_plot(self) -> Path:
        """Create main embedding plot"""
        print(" Creating main embedding visualization...")
        
        # Determine plot name based on embedding type
        if self.experiment.results['embedding_type'] == 'whole_graph':
            plot_name = f"{self.data.y.max().item() + 1}classes_whole_graph.png"
            title = f"{self.experiment.results['dataset']['name']} Whole Graph Embedding"
        else:
            plot_name = f"{self.data.y.max().item() + 1}classes_patched_l2g.png"
            title = f"{self.experiment.results['dataset']['name']} Patched L2G Embedding ({self.experiment.results['num_patches']} patches)"
        
        output_file = self.output_dir / plot_name
        self.plot_embedding_datashader(self.embedding, self.data.y, output_file, title)
        
        return output_file
    
    def create_individual_patch_plots(self) -> List[Path]:
        """Create individual patch visualizations"""
        if self.patches is None:
            return []
            
        print("\\n Creating individual patch visualizations...")
        
        patch_dir = self.output_dir / "individual_patches"
        patch_dir.mkdir(exist_ok=True)
        
        patch_files = []
        
        for i, patch in enumerate(self.patches):
            if not hasattr(patch, 'coordinates') or patch.coordinates is None:
                continue
                
            # Get patch nodes and labels
            patch_nodes = torch.tensor(patch.nodes, dtype=torch.long)
            patch_labels = self.data.y[patch_nodes]
            
            # Create output file
            patch_file = patch_dir / f"patch_{i+1:02d}_embedding.png"
            
            # Use smaller DPI for individual patches
            original_dpi = self.plot_params['dpi']
            self.plot_params['dpi'] = self.plot_params['individual_patch_dpi']
            
            self.plot_embedding_datashader(patch.coordinates, patch_labels, patch_file)
            patch_files.append(patch_file)
            
            # Restore original DPI
            self.plot_params['dpi'] = original_dpi
        
        print(f" Individual patch plots saved to {patch_dir}/")
        return patch_files
    
    def create_patch_grid_overview(self) -> Optional[Path]:
        """Create grid overview of all patches"""
        if self.patches is None:
            return None
            
        print("Creating patch grid overview...")
        
        patch_dir = self.output_dir / "individual_patches"
        patch_dir.mkdir(exist_ok=True)
        
        # Calculate grid dimensions
        n_patches = len([p for p in self.patches if hasattr(p, 'coordinates') and p.coordinates is not None])
        n_cols = min(5, n_patches)
        n_rows = (n_patches + n_cols - 1) // n_cols
        
        grid_dpi = self.plot_params['grid_dpi']
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2), dpi=grid_dpi)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Generate consistent colors
        all_labels = self.data.y.cpu().numpy() if hasattr(self.data.y, 'cpu') else self.data.y
        num_classes = len(np.unique(all_labels[all_labels >= 0]))
        colors = sns.color_palette('husl', num_classes)
        
        patch_idx = 0
        for i, patch in enumerate(self.patches):
            if not hasattr(patch, 'coordinates') or patch.coordinates is None:
                continue
                
            if patch_idx >= len(axes):
                break
                
            ax = axes[patch_idx]
            
            # Get patch data
            patch_nodes = torch.tensor(patch.nodes, dtype=torch.long)
            patch_labels = self.data.y[patch_nodes]
            patch_labels_np = patch_labels.cpu().numpy() if hasattr(patch_labels, 'cpu') else patch_labels
            
            # Apply UMAP
            if patch.coordinates.shape[0] > 1:
                try:
                    umap_coords = umap.UMAP(min_dist=0.0, metric='euclidean', verbose=False).fit_transform(patch.coordinates)
                    
                    # Plot each class
                    for class_idx in np.unique(patch_labels_np):
                        if class_idx >= 0:
                            mask = patch_labels_np == class_idx
                            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                                     c=[colors[class_idx]], s=2, alpha=0.7)
                except Exception:
                    ax.text(0.5, 0.5, f"Patch {i+1}\\nUMAP failed", 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f"Patch {i+1}\\nToo few points", 
                       ha='center', va='center', transform=ax.transAxes)
            
            ax.set_title(f"Patch {i+1} ({len(patch.nodes)} nodes)", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, alpha=0.3)
            
            patch_idx += 1
        
        # Hide unused subplots
        for j in range(patch_idx, len(axes)):
            axes[j].set_visible(False)
        
        # Add title and legend
        fig.suptitle('Individual Patch Embeddings (Before Alignment)', fontsize=12, y=0.95)
        
        legend_elements = [plt.scatter([], [], c=[colors[i]], label=f'Class {i}', s=20) 
                          for i in range(num_classes)]
        fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize=8)
        
        plt.tight_layout()
        
        grid_file = patch_dir / "patch_grid_overview.png"
        plt.savefig(grid_file, dpi=grid_dpi, bbox_inches='tight')
        plt.close()
        
        print(f" Patch grid overview saved: {grid_file}")
        return grid_file
    
    def create_all_visualizations(self) -> List[Path]:
        """Create all visualizations and return list of generated files"""
        generated_files = []
        
        # Main embedding plot
        main_plot = self.create_main_plot()
        generated_files.append(main_plot)
        
        # Individual patch plots (if applicable)
        if self.patches is not None:
            patch_files = self.create_individual_patch_plots()
            generated_files.extend(patch_files)
            
            # Patch grid overview
            grid_file = self.create_patch_grid_overview()
            if grid_file:
                generated_files.append(grid_file)
        
        return generated_files


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python run/plot_embeddings.py <config_file>")
        print("Example: python run/plot_embeddings.py configs/cora_patched.yaml")
        print()
        print("Available configs:")
        print("  configs/cora_patched.yaml  - Cora with L2G patches")
        print("  configs/cora_whole.yaml    - Cora whole graph")
        print("  configs/pubmed_patched.yaml - PubMed with L2G patches")
        print("  configs/dgi_patched.yaml   - Cora with DGI method")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    try:
        # Run embedding experiment
        print(" Running embedding experiment...")
        experiment = EmbeddingExperiment(config_file)
        embedding, patches, data = experiment.run_experiment()
        
        # Create visualizations
        print("\\n Creating visualizations...")
        visualizer = EmbeddingVisualizer(experiment)
        generated_files = visualizer.create_all_visualizations()
        
        # Summary
        print("\\n COMPLETE!")
        print("=" * 50)
        print(f"Experiment: {experiment.config['experiment']['name']}")
        print(f"Dataset: {experiment.results['dataset']['name']}")
        print(f"Embedding type: {experiment.results['embedding_type']}")
        print(f"Shape: {embedding.shape}")
        print()
        print("Generated files:")
        for file_path in generated_files:
            print(f"  • {file_path.name}")
        print(f"\\nAll files saved to: {experiment.output_dir}")
        
    except Exception as e:
        print(f" Error: {e}")
        raise


if __name__ == "__main__":
    main()