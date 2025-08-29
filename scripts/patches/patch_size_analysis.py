#!/usr/bin/env python3
"""
Patch Size Analysis

Analyzes average patch sizes for different numbers of patches across Cora and PubMed datasets.
Tests subdivision into 2, 4, 6, 8, 10, 15, and 20 patches.

Usage:
    python patch_size_analysis.py

This will generate patch statistics and save results to CSV and visualization.
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for L2GX imports
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from l2gx.datasets import get_dataset
from l2gx.graphs import TGraph
from l2gx.patch import create_patches

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class PatchSizeAnalyzer:
    """Analyzes patch sizes for different subdivision configurations."""
    
    def __init__(self):
        self.results = []
        
        # Standard patch configuration parameters
        self.patch_config = {
            "clustering_method": "metis",
            "min_overlap": 256,
            "target_overlap": 512,
            "sparsify_method": "resistance",
            "target_patch_degree": 4,
            "use_conductance_weighting": True,
            "verbose": False,
        }
    
    def load_dataset(self, dataset_name: str) -> TGraph:
        """Load and prepare dataset."""
        print(f"Loading {dataset_name} dataset...")
        dataset = get_dataset(dataset_name)
        pg_data = dataset.to("torch-geometric")
        
        data = TGraph(
            edge_index=pg_data.edge_index,
            x=pg_data.x,
            y=pg_data.y,
            num_nodes=pg_data.num_nodes,
        )
        
        print(f"{dataset_name}: {data.num_nodes} nodes, {data.num_edges} edges")
        return data
    
    def analyze_patch_sizes(self, data: TGraph, dataset_name: str, num_patches: int):
        """Analyze patch sizes for a given number of patches."""
        print(f"  Analyzing {dataset_name} with {num_patches} patches...")
        
        try:
            # Generate patches
            patch_graph = create_patches(
                data,
                num_patches=num_patches,
                **self.patch_config
            )
            
            patches = patch_graph.patches
            
            # Calculate statistics
            patch_sizes = [len(patch.nodes) for patch in patches]
            
            stats = {
                "dataset": dataset_name,
                "num_patches": num_patches,
                "total_nodes": data.num_nodes,
                "total_edges": data.num_edges,
                "patches_created": len(patches),
                "min_patch_size": min(patch_sizes),
                "max_patch_size": max(patch_sizes),
                "mean_patch_size": np.mean(patch_sizes),
                "std_patch_size": np.std(patch_sizes),
                "median_patch_size": np.median(patch_sizes),
                "theoretical_size": data.num_nodes / num_patches,
                "size_efficiency": np.mean(patch_sizes) / (data.num_nodes / num_patches)
            }
            
            print(f"    Created {len(patches)} patches, avg size: {stats['mean_patch_size']:.1f} "
                  f"(range: {stats['min_patch_size']}-{stats['max_patch_size']})")
            
            self.results.append(stats)
            
        except Exception as e:
            print(f"    Error with {num_patches} patches: {e}")
            # Still record the attempt
            stats = {
                "dataset": dataset_name,
                "num_patches": num_patches,
                "total_nodes": data.num_nodes,
                "total_edges": data.num_edges,
                "patches_created": 0,
                "min_patch_size": 0,
                "max_patch_size": 0,
                "mean_patch_size": 0,
                "std_patch_size": 0,
                "median_patch_size": 0,
                "theoretical_size": data.num_nodes / num_patches,
                "size_efficiency": 0,
                "error": str(e)
            }
            self.results.append(stats)
    
    def run_analysis(self):
        """Run complete patch size analysis."""
        print("=" * 80)
        print("PATCH SIZE ANALYSIS")
        print("=" * 80)
        
        # Test configurations
        patch_counts = [2, 4, 6, 8, 10, 15, 20]
        datasets = ["Cora", "PubMed"]
        
        for dataset_name in datasets:
            print(f"\n{dataset_name.upper()} DATASET ANALYSIS")
            print("=" * 50)
            
            # Load dataset
            data = self.load_dataset(dataset_name)
            
            # Test different patch counts
            for num_patches in patch_counts:
                self.analyze_patch_sizes(data, dataset_name, num_patches)
        
        print(f"\nAnalysis complete! Analyzed {len(self.results)} configurations.")
        
        return self.results
    
    def create_summary_table(self):
        """Create summary table of results."""
        if not self.results:
            print("No results to summarize.")
            return None
        
        df = pd.DataFrame(self.results)
        
        # Create pivot table for easy comparison
        summary_table = df.pivot_table(
            index='num_patches',
            columns='dataset',
            values=['mean_patch_size', 'std_patch_size', 'size_efficiency'],
            aggfunc='first'
        )
        
        print("\n" + "=" * 80)
        print("PATCH SIZE SUMMARY")
        print("=" * 80)
        
        # Print key statistics
        for dataset in ['Cora', 'PubMed']:
            dataset_data = df[df['dataset'] == dataset]
            if len(dataset_data) > 0:
                print(f"\n{dataset} Dataset:")
                print(f"  Total nodes: {dataset_data.iloc[0]['total_nodes']:,}")
                print(f"  Total edges: {dataset_data.iloc[0]['total_edges']:,}")
                print(f"  Patch configurations tested: {len(dataset_data)}")
        
        print(f"\nAverage Patch Sizes:")
        print("Patches | Cora Avg Size | PubMed Avg Size | Cora Theoretical | PubMed Theoretical")
        print("-" * 80)
        
        for num_patches in sorted(df['num_patches'].unique()):
            cora_data = df[(df['dataset'] == 'Cora') & (df['num_patches'] == num_patches)]
            pubmed_data = df[(df['dataset'] == 'PubMed') & (df['num_patches'] == num_patches)]
            
            cora_avg = cora_data.iloc[0]['mean_patch_size'] if len(cora_data) > 0 else 0
            pubmed_avg = pubmed_data.iloc[0]['mean_patch_size'] if len(pubmed_data) > 0 else 0
            cora_theo = cora_data.iloc[0]['theoretical_size'] if len(cora_data) > 0 else 0
            pubmed_theo = pubmed_data.iloc[0]['theoretical_size'] if len(pubmed_data) > 0 else 0
            
            print(f"{num_patches:7d} | {cora_avg:12.1f} | {pubmed_avg:14.1f} | "
                  f"{cora_theo:15.1f} | {pubmed_theo:17.1f}")
        
        return df
    
    def create_visualizations(self, df):
        """Create visualizations of patch size analysis."""
        if df is None or len(df) == 0:
            print("No data to visualize.")
            return
        
        print("\nCreating visualizations...")
        
        # Set up the plot style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Average patch size vs number of patches
        ax1 = axes[0, 0]
        for dataset in df['dataset'].unique():
            data = df[df['dataset'] == dataset]
            ax1.plot(data['num_patches'], data['mean_patch_size'], 
                    marker='o', linewidth=2, label=dataset)
            
            # Add theoretical line
            ax1.plot(data['num_patches'], data['theoretical_size'], 
                    linestyle='--', alpha=0.6, label=f'{dataset} (theoretical)')
        
        ax1.set_xlabel('Number of Patches')
        ax1.set_ylabel('Average Patch Size (nodes)')
        ax1.set_title('Average Patch Size vs Number of Patches')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Size efficiency
        ax2 = axes[0, 1]
        for dataset in df['dataset'].unique():
            data = df[df['dataset'] == dataset]
            ax2.plot(data['num_patches'], data['size_efficiency'], 
                    marker='s', linewidth=2, label=dataset)
        
        ax2.set_xlabel('Number of Patches')
        ax2.set_ylabel('Size Efficiency (actual/theoretical)')
        ax2.set_title('Patch Size Efficiency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect efficiency')
        
        # Plot 3: Standard deviation of patch sizes
        ax3 = axes[1, 0]
        for dataset in df['dataset'].unique():
            data = df[df['dataset'] == dataset]
            ax3.plot(data['num_patches'], data['std_patch_size'], 
                    marker='^', linewidth=2, label=dataset)
        
        ax3.set_xlabel('Number of Patches')
        ax3.set_ylabel('Standard Deviation of Patch Sizes')
        ax3.set_title('Patch Size Variability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Min/Max patch sizes
        ax4 = axes[1, 1]
        width = 0.35
        x = np.arange(len(df[df['dataset'] == 'Cora']['num_patches'].unique()))
        
        for i, dataset in enumerate(df['dataset'].unique()):
            data = df[df['dataset'] == dataset].sort_values('num_patches')
            ax4.bar(x + i*width, data['max_patch_size'], width, 
                   alpha=0.7, label=f'{dataset} Max', bottom=data['min_patch_size'])
            ax4.bar(x + i*width, data['min_patch_size'], width, 
                   alpha=0.9, label=f'{dataset} Min')
        
        ax4.set_xlabel('Number of Patches')
        ax4.set_ylabel('Patch Size (nodes)')
        ax4.set_title('Min/Max Patch Sizes')
        ax4.set_xticks(x + width/2)
        ax4.set_xticklabels(sorted(df['num_patches'].unique()))
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = Path(__file__).parent / "patch_size_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to: {output_path}")
    
    def save_results(self, df):
        """Save results to CSV file."""
        if df is None:
            return
        
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        
        # Save full results
        csv_path = output_dir / "patch_size_analysis.csv"
        df.to_csv(csv_path, index=False)
        
        # Create summary CSV
        summary_rows = []
        for num_patches in sorted(df['num_patches'].unique()):
            row = {"num_patches": num_patches}
            
            for dataset in ['Cora', 'PubMed']:
                data = df[(df['dataset'] == dataset) & (df['num_patches'] == num_patches)]
                if len(data) > 0:
                    row[f'{dataset.lower()}_avg_size'] = data.iloc[0]['mean_patch_size']
                    row[f'{dataset.lower()}_std_size'] = data.iloc[0]['std_patch_size']
                    row[f'{dataset.lower()}_theoretical'] = data.iloc[0]['theoretical_size']
                    row[f'{dataset.lower()}_efficiency'] = data.iloc[0]['size_efficiency']
                else:
                    row[f'{dataset.lower()}_avg_size'] = 0
                    row[f'{dataset.lower()}_std_size'] = 0
                    row[f'{dataset.lower()}_theoretical'] = 0
                    row[f'{dataset.lower()}_efficiency'] = 0
            
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_dir / "patch_size_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\nResults saved to:")
        print(f"  Full results: {csv_path}")
        print(f"  Summary: {summary_path}")


def main():
    """Main function."""
    try:
        analyzer = PatchSizeAnalyzer()
        
        # Run the analysis
        results = analyzer.run_analysis()
        
        # Create summary and visualizations
        df = analyzer.create_summary_table()
        analyzer.create_visualizations(df)
        analyzer.save_results(df)
        
        print("\n" + "=" * 80)
        print("PATCH SIZE ANALYSIS COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()