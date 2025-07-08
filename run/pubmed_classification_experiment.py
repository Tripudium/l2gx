#!/usr/bin/env python3
"""
PubMed Node Classification Experiment: VGAE vs Patched VGAE

This experiment compares the classification performance of:
1. VGAE on the whole graph
2. Patched VGAE with 20 patches

Across embedding dimensions: 2, 4, 8, 16, 32, 64, 128
With 10 repetitions per configuration.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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
        'figure.figsize': (12, 8),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'lines.linewidth': 2,
        'lines.markersize': 8
    })

def load_pubmed_dataset():
    """Load and analyze the PubMed dataset."""
    
    print("📚 LOADING PUBMED DATASET")
    print("=" * 50)
    
    try:
        dataset = get_dataset("PubMed")
        data = dataset.to("torch-geometric")
        
        print(f"Dataset loaded successfully!")
        print(f"  Nodes: {data.num_nodes:,}")
        print(f"  Edges: {data.num_edges:,}")
        print(f"  Features: {data.x.shape[1]} dimensions")
        print(f"  Classes: {data.y.max().item() + 1}")
        print(f"  Average degree: {(2 * data.num_edges / data.num_nodes):.1f}")
        
        # Class distribution
        labels = data.y.cpu().numpy()
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\nClass distribution:")
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(labels)) * 100
            print(f"  Class {label}: {count:,} nodes ({percentage:.1f}%)")
        
        return data, labels
        
    except Exception as e:
        print(f"Error loading PubMed dataset: {e}")
        print("Make sure the dataset is available in L2GX")
        return None, None

def create_train_test_split(labels, test_size=0.3, random_state=42):
    """Create train/test split for node classification."""
    
    node_indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        node_indices, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels
    )
    
    print(f"Train/test split:")
    print(f"  Training nodes: {len(train_idx):,} ({(1-test_size)*100:.0f}%)")
    print(f"  Test nodes: {len(test_idx):,} ({test_size*100:.0f}%)")
    
    return train_idx, test_idx

def run_single_experiment(data, labels, train_idx, test_idx, method, embedding_dim, 
                         method_params, random_seed):
    """Run a single experiment for one method and embedding dimension."""
    
    np.random.seed(random_seed)
    
    try:
        # Create embedder
        if method == 'vgae':
            embedder = get_embedding(
                'vgae',
                embedding_dim=embedding_dim,
                epochs=200,
                lr=0.01,
                **method_params
            )
        elif method == 'patched_vgae':
            embedder = get_embedding(
                'patched',
                embedding_method='vgae',
                embedding_dim=embedding_dim,
                epochs=200,
                lr=0.01,
                **method_params
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
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
        
        return {
            'success': True,
            'accuracy': accuracy,
            'embedding_time': embedding_time,
            'classification_time': classification_time,
            'total_time': embedding_time + classification_time
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'accuracy': 0.0,
            'embedding_time': 0.0,
            'classification_time': 0.0,
            'total_time': 0.0
        }

def run_full_experiment():
    """Run the complete experiment."""
    
    print("🧪 PUBMED NODE CLASSIFICATION EXPERIMENT")
    print("=" * 80)
    print("Comparing VGAE vs Patched VGAE across multiple embedding dimensions")
    
    # Load dataset
    data, labels = load_pubmed_dataset()
    if data is None:
        return None
    
    # Create train/test split
    print(f"\n📊 CREATING TRAIN/TEST SPLIT")
    print("=" * 50)
    train_idx, test_idx = create_train_test_split(labels, test_size=0.3, random_state=42)
    
    # Experimental setup
    embedding_dimensions = [2, 4, 8, 16, 32, 64, 128]
    n_repetitions = 10
    random_seeds = [42 + i for i in range(n_repetitions)]
    
    methods = {
        'vgae': {
            'name': 'VGAE (Whole Graph)',
            'params': {}
        },
        'patched_vgae': {
            'name': 'Patched VGAE (20 patches)',
            'params': {
                'num_patches': 20,
                'min_overlap': 256,
                'target_overlap': 512,  # target_overlap should be larger than min_overlap
                'target_patch_degree': 5,
                'clustering_method': 'metis',
                'alignment_method': 'l2g',
                'enable_scaling': False,
                'verbose': False
            }
        }
    }
    
    print(f"\n⚙️  EXPERIMENTAL SETUP")
    print("=" * 50)
    print(f"Embedding dimensions: {embedding_dimensions}")
    print(f"Repetitions per config: {n_repetitions}")
    print(f"Total experiments: {len(methods) * len(embedding_dimensions) * n_repetitions}")
    print(f"Methods:")
    for method_key, method_info in methods.items():
        print(f"  - {method_info['name']}")
        if method_info['params']:
            for param, value in method_info['params'].items():
                print(f"    {param}: {value}")
    
    # Run experiments
    print(f"\n🏃 RUNNING EXPERIMENTS")
    print("=" * 50)
    
    results = []
    total_experiments = len(methods) * len(embedding_dimensions) * n_repetitions
    experiment_count = 0
    
    with tqdm(total=total_experiments, desc="Running experiments") as pbar:
        for method_key, method_info in methods.items():
            for embedding_dim in embedding_dimensions:
                for rep in range(n_repetitions):
                    experiment_count += 1
                    random_seed = random_seeds[rep]
                    
                    pbar.set_description(f"{method_info['name']} | Dim={embedding_dim} | Rep={rep+1}")
                    
                    result = run_single_experiment(
                        data=data,
                        labels=labels,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        method=method_key,
                        embedding_dim=embedding_dim,
                        method_params=method_info['params'],
                        random_seed=random_seed
                    )
                    
                    result.update({
                        'method': method_key,
                        'method_name': method_info['name'],
                        'embedding_dim': embedding_dim,
                        'repetition': rep + 1,
                        'random_seed': random_seed
                    })
                    
                    results.append(result)
                    pbar.update(1)
                    
                    # Print progress for failed experiments
                    if not result['success']:
                        tqdm.write(f"❌ Failed: {method_key} dim={embedding_dim} rep={rep+1}: {result['error']}")
    
    return results, data, labels, train_idx, test_idx

def analyze_results(results):
    """Analyze and summarize the experimental results."""
    
    print(f"\n📈 ANALYZING RESULTS")
    print("=" * 50)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Success rate
    success_rate = df['success'].mean() * 100
    print(f"Overall success rate: {success_rate:.1f}%")
    
    # Failed experiments
    failed_df = df[~df['success']]
    if len(failed_df) > 0:
        print(f"\nFailed experiments: {len(failed_df)}")
        for _, row in failed_df.iterrows():
            print(f"  {row['method_name']} | Dim={row['embedding_dim']} | Rep={row['repetition']}: {row['error']}")
    
    # Filter successful experiments
    successful_df = df[df['success']].copy()
    
    if len(successful_df) == 0:
        print("❌ No successful experiments!")
        return None
    
    # Aggregate statistics
    print(f"\n📊 PERFORMANCE SUMMARY")
    print("=" * 50)
    
    summary_stats = successful_df.groupby(['method_name', 'embedding_dim']).agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'total_time': ['mean', 'std'],
        'repetition': 'count'
    }).round(4)
    
    print("Average accuracy by method and embedding dimension:")
    for method in successful_df['method_name'].unique():
        print(f"\n{method}:")
        method_data = successful_df[successful_df['method_name'] == method]
        for dim in sorted(method_data['embedding_dim'].unique()):
            dim_data = method_data[method_data['embedding_dim'] == dim]
            if len(dim_data) > 0:
                mean_acc = dim_data['accuracy'].mean()
                std_acc = dim_data['accuracy'].std()
                n_reps = len(dim_data)
                print(f"  Dim {dim:3d}: {mean_acc:.4f} ± {std_acc:.4f} ({n_reps} reps)")
    
    return successful_df

def create_visualization(results_df):
    """Create comprehensive visualization of results."""
    
    print(f"\n🎨 CREATING VISUALIZATIONS")
    print("=" * 50)
    
    setup_plot_style()
    
    # Prepare data for plotting
    summary_data = results_df.groupby(['method_name', 'embedding_dim']).agg({
        'accuracy': ['mean', 'std'],
        'total_time': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    summary_data.columns = ['method_name', 'embedding_dim', 'accuracy_mean', 'accuracy_std', 'time_mean', 'time_std']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color palette
    colors = sns.color_palette("Set1", n_colors=len(summary_data['method_name'].unique()))
    method_colors = dict(zip(summary_data['method_name'].unique(), colors))
    
    # Plot 1: Accuracy vs Embedding Dimension (with error bars)
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
            color=method_colors[method]
        )
    
    ax1.set_xlabel('Embedding Dimension')
    ax1.set_ylabel('Classification Accuracy')
    ax1.set_title('Classification Accuracy vs Embedding Dimension')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks([2, 4, 8, 16, 32, 64, 128])
    ax1.set_xticklabels(['2', '4', '8', '16', '32', '64', '128'])
    
    # Plot 2: Training Time vs Embedding Dimension
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
            color=method_colors[method]
        )
    
    ax2.set_xlabel('Embedding Dimension')
    ax2.set_ylabel('Total Time (seconds)')
    ax2.set_title('Training Time vs Embedding Dimension')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.set_xticks([2, 4, 8, 16, 32, 64, 128])
    ax2.set_xticklabels(['2', '4', '8', '16', '32', '64', '128'])
    ax2.set_yscale('log')
    
    # Plot 3: Box plot of accuracies
    ax3 = axes[1, 0]
    
    # Prepare data for box plot
    box_data = []
    box_labels = []
    box_colors = []
    
    for dim in sorted(results_df['embedding_dim'].unique()):
        for method in results_df['method_name'].unique():
            method_data = results_df[
                (results_df['embedding_dim'] == dim) & 
                (results_df['method_name'] == method)
            ]
            if len(method_data) > 0:
                box_data.append(method_data['accuracy'].values)
                box_labels.append(f'{method}\nDim {dim}')
                box_colors.append(method_colors[method])
    
    box_plot = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Classification Accuracy')
    ax3.set_title('Accuracy Distribution by Method and Dimension')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Accuracy improvement (Patched vs VGAE)
    ax4 = axes[1, 1]
    
    # Calculate improvement
    vgae_data = summary_data[summary_data['method_name'].str.contains('VGAE \\(Whole')]
    patched_data = summary_data[summary_data['method_name'].str.contains('Patched')]
    
    if len(vgae_data) > 0 and len(patched_data) > 0:
        improvement_data = []
        dims = []
        
        for dim in sorted(vgae_data['embedding_dim'].unique()):
            vgae_acc = vgae_data[vgae_data['embedding_dim'] == dim]['accuracy_mean'].iloc[0]
            patched_acc = patched_data[patched_data['embedding_dim'] == dim]['accuracy_mean'].iloc[0]
            improvement = patched_acc - vgae_acc
            improvement_data.append(improvement)
            dims.append(dim)
        
        bars = ax4.bar(dims, improvement_data, color='steelblue', alpha=0.7)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Embedding Dimension')
        ax4.set_ylabel('Accuracy Improvement\n(Patched - VGAE)')
        ax4.set_title('Patched VGAE Improvement over VGAE')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log', base=2)
        ax4.set_xticks([2, 4, 8, 16, 32, 64, 128])
        ax4.set_xticklabels(['2', '4', '8', '16', '32', '64', '128'])
        
        # Add value labels on bars
        for bar, improvement in zip(bars, improvement_data):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{improvement:+.3f}',
                    ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('pubmed_classification_experiment.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'pubmed_classification_experiment.png'")
    plt.show()

def save_results(results, filename='pubmed_experiment_results.json'):
    """Save results to JSON file."""
    
    print(f"\n💾 SAVING RESULTS")
    print("=" * 50)
    
    # Convert to serializable format
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
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to '{filename}'")

def main():
    """Main function to run the complete experiment."""
    
    print("🔬 PUBMED NODE CLASSIFICATION EXPERIMENT")
    print("=" * 90)
    print("This experiment compares VGAE vs Patched VGAE on PubMed dataset")
    print("across embedding dimensions 2, 4, 8, 16, 32, 64, 128 with 10 repetitions each.")
    
    # Run the experiment
    results = run_full_experiment()
    if results is None:
        print("❌ Experiment failed to complete")
        return
    
    results_list, data, labels, train_idx, test_idx = results
    
    # Save raw results
    save_results(results_list)
    
    # Analyze results
    results_df = analyze_results(results_list)
    if results_df is None:
        print("❌ Analysis failed")
        return
    
    # Create visualizations
    create_visualization(results_df)
    
    # Final summary
    print(f"\n✅ EXPERIMENT COMPLETE!")
    print("=" * 50)
    print("Key outputs:")
    print("• pubmed_classification_experiment.png - Main visualization")
    print("• pubmed_experiment_results.json - Raw results data")
    print("\nThe experiment compared classification accuracy across embedding dimensions")
    print("for VGAE (whole graph) vs Patched VGAE (20 patches) on PubMed dataset.")

if __name__ == "__main__":
    main()