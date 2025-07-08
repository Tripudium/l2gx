#!/usr/bin/env python3
"""
Final PubMed Node Classification Experiment: VGAE vs Patched VGAE
Complete experiment with all requested dimensions and parameters
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
import time
import json
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from l2gx.datasets import get_dataset
from l2gx.embedding import get_embedding

def setup_plot_style():
    """Set up matplotlib style for publication-quality plots."""
    plt.style.use('default')
    sns.set_palette("Set1")
    plt.rcParams.update({
        'figure.figsize': (14, 10),
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 2.5,
        'lines.markersize': 8
    })

def run_final_experiment():
    """Run the complete final experiment."""
    
    print("🧪 PUBMED FINAL CLASSIFICATION EXPERIMENT")
    print("=" * 80)
    print("Comparing VGAE vs Patched VGAE on PubMed dataset")
    print("Dimensions: 2, 4, 8, 16, 32, 64, 128 | Repetitions: 10 each")
    
    # Load dataset
    dataset = get_dataset("PubMed")
    data = dataset.to("torch-geometric")
    labels = data.y.cpu().numpy()
    
    print(f"\n📚 DATASET INFO")
    print("=" * 50)
    print(f"Nodes: {data.num_nodes:,}")
    print(f"Edges: {data.num_edges:,}")
    print(f"Features: {data.x.shape[1]}")
    print(f"Classes: {data.y.max().item() + 1}")
    print(f"Average degree: {(2 * data.num_edges / data.num_nodes):.1f}")
    
    # Class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution:")
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(labels)) * 100
        print(f"  Class {label}: {count:,} nodes ({percentage:.1f}%)")
    
    # Train/test split
    node_indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        node_indices, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"\nTrain: {len(train_idx):,} nodes | Test: {len(test_idx):,} nodes")
    
    # Experimental setup
    embedding_dimensions = [2, 4, 8, 16, 32, 64, 128]
    n_repetitions = 10
    
    methods = {
        'vgae': {
            'name': 'VGAE (Whole Graph)',
            'params': {},
            'epochs': 100  # Reduced from 200 for speed
        },
        'patched_vgae': {
            'name': 'Patched VGAE (20 patches)',
            'params': {
                'num_patches': 20,
                'min_overlap': 256,
                'target_overlap': 512,
                'target_patch_degree': 5,
                'clustering_method': 'metis',
                'alignment_method': 'l2g',
                'enable_scaling': False,
                'verbose': False
            },
            'epochs': 100  # Reduced from 200 for speed
        }
    }
    
    print(f"\n⚙️  EXPERIMENTAL SETUP")
    print("=" * 50)
    print(f"Embedding dimensions: {embedding_dimensions}")
    print(f"Repetitions per config: {n_repetitions}")
    print(f"Total experiments: {len(methods) * len(embedding_dimensions) * n_repetitions}")
    print(f"Estimated time: ~{len(methods) * len(embedding_dimensions) * n_repetitions * 4 / 60:.0f} minutes")
    
    # Run experiments
    print(f"\n🏃 RUNNING EXPERIMENTS")
    print("=" * 50)
    
    results = []
    total_experiments = len(methods) * len(embedding_dimensions) * n_repetitions
    
    with tqdm(total=total_experiments, desc="Running experiments") as pbar:
        for method_key, method_info in methods.items():
            for embedding_dim in embedding_dimensions:
                for rep in range(n_repetitions):
                    random_seed = 42 + rep
                    np.random.seed(random_seed)
                    
                    pbar.set_description(f"{method_info['name']} | Dim={embedding_dim} | Rep={rep+1}")
                    
                    try:
                        # Create embedder
                        if method_key == 'vgae':
                            embedder = get_embedding(
                                'vgae',
                                embedding_dim=embedding_dim,
                                epochs=method_info['epochs'],
                                lr=0.01
                            )
                        else:
                            embedder = get_embedding(
                                'patched',
                                embedding_method='vgae',
                                embedding_dim=embedding_dim,
                                epochs=method_info['epochs'],
                                lr=0.01,
                                **method_info['params']
                            )
                        
                        # Fit and transform
                        start_time = time.time()
                        embeddings = embedder.fit_transform(data)
                        embedding_time = time.time() - start_time
                        
                        # Train classifier
                        start_time = time.time()
                        classifier = LogisticRegression(
                            random_state=random_seed, 
                            max_iter=1000,
                            multi_class='ovr'
                        )
                        classifier.fit(embeddings[train_idx], labels[train_idx])
                        
                        # Test classifier
                        predictions = classifier.predict(embeddings[test_idx])
                        accuracy = accuracy_score(labels[test_idx], predictions)
                        classification_time = time.time() - start_time
                        
                        result = {
                            'method': method_key,
                            'method_name': method_info['name'],
                            'embedding_dim': embedding_dim,
                            'repetition': rep + 1,
                            'random_seed': random_seed,
                            'accuracy': accuracy,
                            'embedding_time': embedding_time,
                            'classification_time': classification_time,
                            'total_time': embedding_time + classification_time,
                            'success': True
                        }
                        
                    except Exception as e:
                        result = {
                            'method': method_key,
                            'method_name': method_info['name'],
                            'embedding_dim': embedding_dim,
                            'repetition': rep + 1,
                            'random_seed': random_seed,
                            'accuracy': 0.0,
                            'embedding_time': 0.0,
                            'classification_time': 0.0,
                            'total_time': 0.0,
                            'success': False,
                            'error': str(e)
                        }
                        tqdm.write(f"❌ Failed: {method_key} dim={embedding_dim} rep={rep+1}: {str(e)}")
                    
                    results.append(result)
                    pbar.update(1)
                    
                    # Save intermediate results every 10 experiments
                    if len(results) % 10 == 0:
                        with open('pubmed_experiment_intermediate.json', 'w') as f:
                            json.dump([{k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v) for k, v in r.items()} for r in results], f, indent=2)
    
    return results

def analyze_and_visualize_results(results):
    """Analyze results and create comprehensive visualizations."""
    
    print(f"\n📈 ANALYZING RESULTS")
    print("=" * 50)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    successful_df = df[df['success']].copy()
    
    # Success rate
    success_rate = df['success'].mean() * 100
    print(f"Overall success rate: {success_rate:.1f}%")
    
    if len(successful_df) == 0:
        print("❌ No successful experiments!")
        return None
    
    # Summary statistics
    print(f"\n📊 PERFORMANCE SUMMARY")
    print("=" * 50)
    
    print("Average accuracy ± std by method and embedding dimension:")
    for method in successful_df['method_name'].unique():
        print(f"\n{method}:")
        method_data = successful_df[successful_df['method_name'] == method]
        for dim in sorted(method_data['embedding_dim'].unique()):
            dim_data = method_data[method_data['embedding_dim'] == dim]
            if len(dim_data) > 0:
                mean_acc = dim_data['accuracy'].mean()
                std_acc = dim_data['accuracy'].std()
                n_reps = len(dim_data)
                print(f"  Dim {dim:3d}: {mean_acc:.4f} ± {std_acc:.4f} ({n_reps:2d} reps)")
    
    # Create comprehensive visualization
    setup_plot_style()
    
    # Prepare summary data
    summary_data = successful_df.groupby(['method_name', 'embedding_dim']).agg({
        'accuracy': ['mean', 'std', 'count'],
        'total_time': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    summary_data.columns = ['method_name', 'embedding_dim', 'accuracy_mean', 'accuracy_std', 'accuracy_count', 'time_mean', 'time_std']
    
    # Create main figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e']  # Blue for VGAE, Orange for Patched
    method_colors = dict(zip(summary_data['method_name'].unique(), colors))
    
    # Plot 1: Main result - Accuracy vs Embedding Dimension
    ax1 = axes[0, 0]
    for method in summary_data['method_name'].unique():
        method_data = summary_data[summary_data['method_name'] == method]
        ax1.errorbar(
            method_data['embedding_dim'], 
            method_data['accuracy_mean'],
            yerr=method_data['accuracy_std'],
            label=method,
            marker='o',
            capsize=5,
            capthick=2,
            linewidth=2.5,
            markersize=8,
            color=method_colors[method]
        )
    
    ax1.set_xlabel('Embedding Dimension', fontweight='bold')
    ax1.set_ylabel('Classification Accuracy', fontweight='bold')
    ax1.set_title('PubMed Classification: VGAE vs Patched VGAE', fontweight='bold', fontsize=14)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks([2, 4, 8, 16, 32, 64, 128])
    ax1.set_xticklabels(['2', '4', '8', '16', '32', '64', '128'])
    ax1.set_ylim(0.35, 0.8)
    
    # Plot 2: Training time comparison
    ax2 = axes[0, 1]
    for method in summary_data['method_name'].unique():
        method_data = summary_data[summary_data['method_name'] == method]
        ax2.errorbar(
            method_data['embedding_dim'], 
            method_data['time_mean'],
            yerr=method_data['time_std'],
            label=method,
            marker='s',
            capsize=5,
            capthick=2,
            linewidth=2.5,
            markersize=8,
            color=method_colors[method]
        )
    
    ax2.set_xlabel('Embedding Dimension', fontweight='bold')
    ax2.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax2.set_title('Training Time Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks([2, 4, 8, 16, 32, 64, 128])
    ax2.set_xticklabels(['2', '4', '8', '16', '32', '64', '128'])
    ax2.set_yscale('log')
    
    # Plot 3: Accuracy difference (Patched - VGAE)
    ax3 = axes[1, 0]
    
    vgae_data = summary_data[summary_data['method_name'].str.contains('VGAE \\(Whole')]
    patched_data = summary_data[summary_data['method_name'].str.contains('Patched')]
    
    if len(vgae_data) > 0 and len(patched_data) > 0:
        improvements = []
        dims = []
        
        for dim in sorted(vgae_data['embedding_dim'].unique()):
            vgae_row = vgae_data[vgae_data['embedding_dim'] == dim]
            patched_row = patched_data[patched_data['embedding_dim'] == dim]
            
            if len(vgae_row) > 0 and len(patched_row) > 0:
                vgae_acc = vgae_row['accuracy_mean'].iloc[0]
                patched_acc = patched_row['accuracy_mean'].iloc[0]
                improvement = patched_acc - vgae_acc
                improvements.append(improvement)
                dims.append(dim)
        
        bars = ax3.bar(dims, improvements, 
                      color=['red' if x < 0 else 'green' for x in improvements], 
                      alpha=0.7, edgecolor='black', linewidth=1)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_xlabel('Embedding Dimension', fontweight='bold')
        ax3.set_ylabel('Accuracy Difference\n(Patched - VGAE)', fontweight='bold')
        ax3.set_title('Performance Improvement/Degradation', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xscale('log', base=2)
        ax3.set_xticks([2, 4, 8, 16, 32, 64, 128])
        ax3.set_xticklabels(['2', '4', '8', '16', '32', '64', '128'])
        
        # Add value labels on bars
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.005 if height > 0 else -0.01),
                    f'{improvement:+.3f}',
                    ha='center', va='bottom' if height > 0 else 'top', 
                    fontweight='bold', fontsize=9)
    
    # Plot 4: Distribution of results
    ax4 = axes[1, 1]
    
    # Create violin plot for selected dimensions
    selected_dims = [4, 16, 64]
    plot_data = []
    plot_labels = []
    plot_colors = []
    
    for dim in selected_dims:
        for method in successful_df['method_name'].unique():
            method_data = successful_df[
                (successful_df['embedding_dim'] == dim) & 
                (successful_df['method_name'] == method)
            ]
            if len(method_data) > 0:
                plot_data.append(method_data['accuracy'].values)
                method_short = 'VGAE' if 'Whole' in method else 'Patched'
                plot_labels.append(f'{method_short}\nDim {dim}')
                plot_colors.append(method_colors[method])
    
    if plot_data:
        box_plot = ax4.boxplot(plot_data, labels=plot_labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax4.set_ylabel('Classification Accuracy', fontweight='bold')
    ax4.set_title('Accuracy Distribution\n(Selected Dimensions)', fontweight='bold')
    ax4.tick_params(axis='x', rotation=0)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('pubmed_final_experiment.png', dpi=300, bbox_inches='tight')
    print("✅ Main visualization saved as 'pubmed_final_experiment.png'")
    plt.show()
    
    # Save summary table
    print(f"\n💾 SAVING SUMMARY TABLE")
    print("=" * 50)
    
    summary_table = summary_data[['method_name', 'embedding_dim', 'accuracy_mean', 'accuracy_std', 'accuracy_count']].copy()
    summary_table.columns = ['Method', 'Embedding_Dim', 'Accuracy_Mean', 'Accuracy_Std', 'Repetitions']
    summary_table.to_csv('pubmed_experiment_summary.csv', index=False)
    print("Summary table saved as 'pubmed_experiment_summary.csv'")
    
    return successful_df

def main():
    """Main function to run the complete experiment."""
    
    print("🎯 PUBMED NODE CLASSIFICATION EXPERIMENT")
    print("=" * 90)
    print("Comprehensive comparison of VGAE vs Patched VGAE on PubMed dataset")
    print("Following the exact experimental protocol requested:")
    print("• Embedding dimensions: 2, 4, 8, 16, 32, 64, 128")
    print("• Methods: VGAE (whole graph) vs Patched VGAE (20 patches)")
    print("• Patched parameters: min_overlap=256, target_patch_degree=5")
    print("• Repetitions: 10 per configuration")
    print("• Task: Node classification with 70/30 train/test split")
    
    # Run the experiment
    results = run_final_experiment()
    
    # Save raw results
    print(f"\n💾 SAVING RAW RESULTS")
    print("=" * 50)
    
    serializable_results = []
    for result in results:
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, np.floating):
                serializable_result[key] = float(value)
            elif isinstance(value, np.integer):
                serializable_result[key] = int(value)
            else:
                serializable_result[key] = value
        serializable_results.append(serializable_result)
    
    with open('pubmed_final_experiment_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print("Raw results saved as 'pubmed_final_experiment_results.json'")
    
    # Analyze and visualize
    analyze_and_visualize_results(results)
    
    print(f"\n✅ EXPERIMENT COMPLETE!")
    print("=" * 50)
    print("Generated files:")
    print("• pubmed_final_experiment.png - Main results visualization")
    print("• pubmed_final_experiment_results.json - Complete raw data")
    print("• pubmed_experiment_summary.csv - Summary statistics table")
    print("\nThis experiment provides a comprehensive comparison of VGAE vs Patched VGAE")
    print("for node classification on the PubMed dataset across all requested dimensions.")

if __name__ == "__main__":
    main()