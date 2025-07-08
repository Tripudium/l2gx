"""
Cora Dataset Experiments Script.

This script demonstrates how to run node reconstruction and classification
experiments on the Cora dataset using the L2GX framework with configuration management.
"""

import os
import sys
import time
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pickle
import json

# Add the parent directory to the path to import l2gx
sys.path.append(str(Path(__file__).parent.parent))

from l2gx.datasets import get_dataset
from l2gx.embedding import get_embedding
from l2gx.patch import generate_patches
from l2gx.align import get_aligner
from l2gx.graphs import TGraph

from config_manager import ConfigManager, Config

# Import evaluation modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class CoraExperimentRunner:
    """Main experiment runner for Cora dataset experiments."""
    
    def __init__(self, config_path: str):
        """
        Initialize experiment runner.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config(config_path)
        self.logger = None
        self.results = {}
        
        # Setup experiment environment
        self._setup_experiment()
    
    def _setup_experiment(self):
        """Setup experiment environment."""
        # Set random seeds
        seed = self.config.experiment.random_seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Create output directories
        self.config_manager.create_output_directories()
        
        # Setup logging
        self.config_manager.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Starting experiment: {self.config.experiment.name}")
        self.logger.info(f"Configuration loaded from: {self.config_manager.config_path}")
        
        # Set device
        if self.config.experiment.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.experiment.device
        
        self.logger.info(f"Using device: {self.device}")
    
    def run_experiments(self) -> Dict[str, Any]:
        """
        Run complete experiment pipeline.
        
        Returns:
            Dictionary containing all experimental results
        """
        self.logger.info("="*60)
        self.logger.info("STARTING CORA DATASET EXPERIMENTS")
        self.logger.info("="*60)
        
        # Load dataset
        dataset, data = self._load_dataset()
        
        # Prepare data splits
        splits = self._prepare_data_splits(data)
        
        # Run multiple experimental runs
        all_results = []
        for run_idx in range(self.config.experiment.num_runs):
            self.logger.info(f"\n--- EXPERIMENTAL RUN {run_idx + 1}/{self.config.experiment.num_runs} ---")
            
            run_results = self._run_single_experiment(dataset, data, splits, run_idx)
            all_results.append(run_results)
        
        # Aggregate results across runs
        final_results = self._aggregate_results(all_results)
        
        # Save results
        self._save_results(final_results)
        
        # Create visualizations
        if self.config.visualization.enabled:
            self._create_visualizations(final_results)
        
        self.logger.info("="*60)
        self.logger.info("EXPERIMENTS COMPLETED SUCCESSFULLY")
        self.logger.info("="*60)
        
        return final_results
    
    def _load_dataset(self) -> Tuple[Any, Any]:
        """Load Cora dataset."""
        self.logger.info("Loading Cora dataset...")
        
        dataset = get_dataset(self.config.dataset.name)
        data = dataset.to("torch-geometric")
        
        self.logger.info(f"Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges")
        self.logger.info(f"Features: {data.x.shape[1]}, Classes: {data.y.unique().numel()}")
        
        return dataset, data
    
    def _prepare_data_splits(self, data) -> Dict[str, np.ndarray]:
        """Prepare train/validation/test splits."""
        self.logger.info("Preparing data splits...")
        
        if self.config.dataset.use_default_splits and hasattr(data, 'train_mask'):
            # Use predefined splits from the dataset
            splits = {
                'train_mask': data.train_mask.numpy(),
                'val_mask': data.val_mask.numpy(),
                'test_mask': data.test_mask.numpy()
            }
            self.logger.info("Using predefined dataset splits")
        else:
            # Create custom splits
            n_nodes = data.num_nodes
            indices = np.arange(n_nodes)
            labels = data.y.numpy()
            
            # Split into train/temp
            train_indices, temp_indices = train_test_split(
                indices, 
                test_size=1-self.config.dataset.train_ratio,
                stratify=labels,
                random_state=self.config.dataset.split_seed
            )
            
            # Split temp into val/test
            val_ratio = self.config.dataset.val_ratio / (self.config.dataset.val_ratio + self.config.dataset.test_ratio)
            val_indices, test_indices = train_test_split(
                temp_indices,
                test_size=1-val_ratio,
                stratify=labels[temp_indices],
                random_state=self.config.dataset.split_seed
            )
            
            # Create masks
            splits = {
                'train_mask': np.isin(indices, train_indices),
                'val_mask': np.isin(indices, val_indices),
                'test_mask': np.isin(indices, test_indices)
            }
            
            self.logger.info(f"Created custom splits: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
        
        return splits
    
    def _run_single_experiment(self, dataset, data, splits, run_idx: int) -> Dict[str, Any]:
        """Run a single experimental run."""
        run_results = {'run_id': run_idx}
        
        # Step 1: Generate patches (if using patch-based approach)
        if self.config.patches.num_patches > 1:
            patches, patch_graph = self._generate_patches(dataset, data)
            run_results['patches'] = {
                'num_patches': len(patches),
                'patch_sizes': [len(p.nodes) for p in patches],
                'avg_patch_size': np.mean([len(p.nodes) for p in patches])
            }
        else:
            patches = None
            patch_graph = None
        
        # Step 2: Generate embeddings
        if patches is not None:
            embeddings = self._generate_patch_embeddings(patches, data)
            aligned_embeddings = self._align_embeddings(patches, embeddings)
            final_embeddings = aligned_embeddings
        else:
            final_embeddings = self._generate_global_embeddings(data)
        
        run_results['embeddings'] = {
            'method': self.config.embedding.method,
            'dimension': final_embeddings.shape[1],
            'shape': final_embeddings.shape
        }
        
        # Step 3: Node reconstruction task
        reconstruction_results = self._run_node_reconstruction(final_embeddings, data, splits)
        run_results['node_reconstruction'] = reconstruction_results
        
        # Step 4: Node classification task
        classification_results = self._run_node_classification(final_embeddings, data, splits)
        run_results['node_classification'] = classification_results
        
        # Save intermediate results if requested
        if self.config.experiment.save_intermediate:
            self._save_intermediate_results(run_results, run_idx)
        
        return run_results
    
    def _generate_patches(self, dataset, data) -> Tuple[List, Any]:
        """Generate graph patches."""
        self.logger.info(f"Generating {self.config.patches.num_patches} patches...")
        
        # Convert to TGraph for patch generation
        tg = TGraph(data.edge_index, edge_attr=data.edge_attr, x=data.x)
        
        # Get patch parameters
        patch_params = self.config_manager.get_patch_params()
        
        patches, patch_graph = generate_patches(tg, **patch_params)
        
        patch_sizes = [len(p.nodes) for p in patches]
        self.logger.info(f"Created {len(patches)} patches, sizes: [{min(patch_sizes)}, {max(patch_sizes)}], avg: {np.mean(patch_sizes):.1f}")
        
        return patches, patch_graph
    
    def _generate_patch_embeddings(self, patches, data) -> List[np.ndarray]:
        """Generate embeddings for each patch."""
        self.logger.info(f"Generating embeddings for {len(patches)} patches...")
        
        embedding_params = self.config_manager.get_embedding_params()
        embeddings = []
        
        for i, patch in enumerate(patches):
            self.logger.info(f"Processing patch {i+1}/{len(patches)} ({len(patch.nodes)} nodes)")
            
            # Extract subgraph for this patch
            patch_nodes = patch.nodes
            
            # Create patch-specific data
            node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(patch_nodes)}
            
            # Get edges within the patch
            edge_index = data.edge_index
            mask = torch.isin(edge_index[0], torch.tensor(patch_nodes)) & torch.isin(edge_index[1], torch.tensor(patch_nodes))
            patch_edges = edge_index[:, mask]
            
            # Remap edge indices to patch-local indices
            patch_edge_index = torch.zeros_like(patch_edges)
            for j, (src, dst) in enumerate(patch_edges.t()):
                patch_edge_index[0, j] = node_mapping[src.item()]
                patch_edge_index[1, j] = node_mapping[dst.item()]
            
            # Create patch data object
            patch_data = type(data)(
                x=data.x[patch_nodes],
                edge_index=patch_edge_index,
                y=data.y[patch_nodes] if hasattr(data, 'y') else None
            )
            
            # Generate embedding for this patch
            embedder = get_embedding(self.config.embedding.method, **embedding_params)
            patch_embedding = embedder.fit_transform(patch_data)
            
            # Convert to numpy if needed
            if torch.is_tensor(patch_embedding):
                patch_embedding = patch_embedding.detach().cpu().numpy()
            
            embeddings.append(patch_embedding)
        
        return embeddings
    
    def _align_embeddings(self, patches, embeddings) -> np.ndarray:
        """Align patch embeddings to create global embedding."""
        self.logger.info("Aligning patch embeddings...")
        
        # Create patch objects with embeddings
        embedding_patches = []
        for patch, embedding in zip(patches, embeddings):
            from l2gx.patch.patches import Patch
            emb_patch = Patch(patch.nodes, embedding)
            embedding_patches.append(emb_patch)
        
        # Get alignment parameters
        alignment_params = self.config_manager.get_alignment_params()
        
        # Create aligner and perform alignment
        aligner = get_aligner(self.config.alignment.method, **alignment_params)
        result = aligner.align_patches(embedding_patches)
        
        # Extract aligned embedding
        aligned_embedding = result.get_aligned_embedding()
        
        self.logger.info(f"Alignment completed. Final embedding shape: {aligned_embedding.shape}")
        
        return aligned_embedding
    
    def _generate_global_embeddings(self, data) -> np.ndarray:
        """Generate global embeddings (non-patch approach)."""
        self.logger.info("Generating global embeddings...")
        
        embedding_params = self.config_manager.get_embedding_params()
        embedder = get_embedding(self.config.embedding.method, **embedding_params)
        
        embeddings = embedder.fit_transform(data)
        
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().cpu().numpy()
        
        self.logger.info(f"Global embeddings generated: {embeddings.shape}")
        
        return embeddings
    
    def _run_node_reconstruction(self, embeddings, data, splits) -> Dict[str, Any]:
        """Run node reconstruction task."""
        self.logger.info("Running node reconstruction task...")
        
        # Use node features as reconstruction targets
        targets = data.x.numpy() if torch.is_tensor(data.x) else data.x
        
        # Split data
        train_mask = splits['train_mask']
        test_mask = splits['test_mask']
        
        X_train = embeddings[train_mask]
        y_train = targets[train_mask]
        X_test = embeddings[test_mask]
        y_test = targets[test_mask]
        
        # Train reconstruction model
        config = self.config.node_reconstruction
        
        if config.reconstruction_method == "linear_decoder":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif config.reconstruction_method == "mlp_decoder":
            from sklearn.neural_network import MLPRegressor
            model = MLPRegressor(
                hidden_layer_sizes=tuple(config.decoder_hidden_dims),
                max_iter=500,
                random_state=self.config.experiment.random_seed
            )
        else:  # autoencoder - use simple linear for now
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        
        # Fit and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Compute metrics
        results = {}
        if "mse" in config.evaluation_metrics:
            results['mse'] = mean_squared_error(y_test, y_pred)
        if "mae" in config.evaluation_metrics:
            results['mae'] = mean_absolute_error(y_test, y_pred)
        if "cosine_similarity" in config.evaluation_metrics:
            from sklearn.metrics.pairwise import cosine_similarity
            cos_sim = np.mean([cosine_similarity([y_test[i]], [y_pred[i]])[0,0] 
                              for i in range(len(y_test))])
            results['cosine_similarity'] = cos_sim
        
        self.logger.info(f"Node reconstruction results: {results}")
        
        return results
    
    def _run_node_classification(self, embeddings, data, splits) -> Dict[str, Any]:
        """Run node classification task."""
        self.logger.info("Running node classification task...")
        
        # Get labels
        labels = data.y.numpy() if torch.is_tensor(data.y) else data.y
        
        # Split data
        train_mask = splits['train_mask']
        val_mask = splits['val_mask']
        test_mask = splits['test_mask']
        
        X_train = embeddings[train_mask]
        y_train = labels[train_mask]
        X_val = embeddings[val_mask]
        y_val = labels[val_mask]
        X_test = embeddings[test_mask]
        y_test = labels[test_mask]
        
        config = self.config.node_classification
        
        # Create classifier
        if config.classifier == "logistic_regression":
            model = LogisticRegression(
                random_state=self.config.experiment.random_seed,
                max_iter=1000
            )
        elif config.classifier == "svm":
            model = SVC(
                kernel=config.svm_kernel,
                C=config.svm_C,
                random_state=self.config.experiment.random_seed
            )
        elif config.classifier == "random_forest":
            model = RandomForestClassifier(
                n_estimators=config.rf_n_estimators,
                max_depth=config.rf_max_depth,
                random_state=self.config.experiment.random_seed
            )
        elif config.classifier == "mlp":
            model = MLPClassifier(
                hidden_layer_sizes=tuple(config.mlp_hidden_dims),
                max_iter=config.mlp_num_epochs,
                learning_rate_init=config.mlp_learning_rate,
                random_state=self.config.experiment.random_seed
            )
        else:
            raise ValueError(f"Unknown classifier: {config.classifier}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Compute metrics
        results = {}
        if "accuracy" in config.evaluation_metrics:
            results['accuracy'] = accuracy_score(y_test, y_pred)
        if "f1_macro" in config.evaluation_metrics:
            results['f1_macro'] = f1_score(y_test, y_pred, average='macro')
        if "f1_micro" in config.evaluation_metrics:
            results['f1_micro'] = f1_score(y_test, y_pred, average='micro')
        if "precision" in config.evaluation_metrics:
            results['precision'] = precision_score(y_test, y_pred, average='macro')
        if "recall" in config.evaluation_metrics:
            results['recall'] = recall_score(y_test, y_pred, average='macro')
        
        self.logger.info(f"Node classification results: {results}")
        
        return results
    
    def _aggregate_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across multiple runs."""
        self.logger.info("Aggregating results across runs...")
        
        aggregated = {
            'config': self.config_manager._config_to_dict(self.config),
            'num_runs': len(all_results),
            'individual_runs': all_results
        }
        
        # Aggregate node reconstruction results
        recon_metrics = []
        for run_result in all_results:
            if 'node_reconstruction' in run_result:
                recon_metrics.append(run_result['node_reconstruction'])
        
        if recon_metrics:
            aggregated['node_reconstruction'] = self._aggregate_metrics(recon_metrics)
        
        # Aggregate node classification results
        class_metrics = []
        for run_result in all_results:
            if 'node_classification' in run_result:
                class_metrics.append(run_result['node_classification'])
        
        if class_metrics:
            aggregated['node_classification'] = self._aggregate_metrics(class_metrics)
        
        return aggregated
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across runs."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        # Get all metric names
        all_metrics = set()
        for metrics in metrics_list:
            all_metrics.update(metrics.keys())
        
        # Compute statistics for each metric
        for metric in all_metrics:
            values = [metrics.get(metric, np.nan) for metrics in metrics_list]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                aggregated[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return aggregated
    
    def _save_intermediate_results(self, results: Dict[str, Any], run_idx: int):
        """Save intermediate results for a single run."""
        output_dir = Path(self.config.experiment.output_dir) / "intermediate"
        output_dir.mkdir(exist_ok=True)
        
        filename = f"run_{run_idx:03d}_results.json"
        with open(output_dir / filename, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._make_json_serializable(results)
            json.dump(json_results, f, indent=2)
    
    def _save_results(self, results: Dict[str, Any]):
        """Save final aggregated results."""
        output_dir = Path(self.config.experiment.output_dir)
        
        # Save as JSON
        json_results = self._make_json_serializable(results)
        with open(output_dir / "final_results.json", 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save as pickle (preserves exact numpy arrays)
        with open(output_dir / "final_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        self.logger.info(f"Results saved to: {output_dir}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def _create_visualizations(self, results: Dict[str, Any]):
        """Create visualizations of the results."""
        self.logger.info("Creating visualizations...")
        
        output_dir = Path(self.config.experiment.output_dir) / "plots"
        
        # Plot results summary
        self._plot_results_summary(results, output_dir)
        
        self.logger.info(f"Visualizations saved to: {output_dir}")
    
    def _plot_results_summary(self, results: Dict[str, Any], output_dir: Path):
        """Plot summary of experimental results."""
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Experiment Results: {self.config.experiment.name}", fontsize=16)
        
        # Plot 1: Node classification metrics
        if 'node_classification' in results:
            ax = axes[0, 0]
            class_results = results['node_classification']
            metrics = []
            values = []
            errors = []
            
            for metric, stats in class_results.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    metrics.append(metric)
                    values.append(stats['mean'])
                    errors.append(stats['std'])
            
            if metrics:
                bars = ax.bar(metrics, values, yerr=errors, capsize=5, alpha=0.7)
                ax.set_title('Node Classification Metrics')
                ax.set_ylabel('Score')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 2: Node reconstruction metrics
        if 'node_reconstruction' in results:
            ax = axes[0, 1]
            recon_results = results['node_reconstruction']
            metrics = []
            values = []
            errors = []
            
            for metric, stats in recon_results.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    metrics.append(metric)
                    values.append(stats['mean'])
                    errors.append(stats['std'])
            
            if metrics:
                bars = ax.bar(metrics, values, yerr=errors, capsize=5, alpha=0.7, color='orange')
                ax.set_title('Node Reconstruction Metrics')
                ax.set_ylabel('Score')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 3: Configuration summary
        ax = axes[1, 0]
        config_text = f"""Configuration Summary:
        
Embedding: {self.config.embedding.method}
Dimension: {self.config.embedding.embedding_dim}
Hidden Dim: {self.config.embedding.hidden_dim}
Epochs: {self.config.embedding.num_epochs}
Learning Rate: {self.config.embedding.learning_rate}

Patches: {self.config.patches.num_patches}
Clustering: {self.config.patches.clustering_method}
Alignment: {self.config.alignment.method}

Runs: {self.config.experiment.num_runs}
Seed: {self.config.experiment.random_seed}"""
        
        ax.text(0.05, 0.95, config_text, transform=ax.transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=10)
        ax.set_title('Experiment Configuration')
        ax.axis('off')
        
        # Plot 4: Performance across runs (if multiple runs)
        ax = axes[1, 1]
        if results['num_runs'] > 1 and 'individual_runs' in results:
            run_numbers = list(range(1, results['num_runs'] + 1))
            
            # Extract accuracy across runs
            accuracies = []
            for run_result in results['individual_runs']:
                if 'node_classification' in run_result and 'accuracy' in run_result['node_classification']:
                    accuracies.append(run_result['node_classification']['accuracy'])
                else:
                    accuracies.append(np.nan)
            
            if not all(np.isnan(accuracies)):
                ax.plot(run_numbers, accuracies, 'o-', linewidth=2, markersize=6)
                ax.set_title('Classification Accuracy Across Runs')
                ax.set_xlabel('Run Number')
                ax.set_ylabel('Accuracy')
                ax.grid(True, alpha=0.3)
                
                # Add mean line
                mean_acc = np.nanmean(accuracies)
                ax.axhline(mean_acc, color='red', linestyle='--', alpha=0.7, 
                          label=f'Mean: {mean_acc:.3f}')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No accuracy data available', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title('Performance Across Runs')
        else:
            ax.text(0.5, 0.5, 'Single run experiment', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Performance Across Runs')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f"results_summary.{self.config.visualization.plot_format}"
        plt.savefig(plot_path, dpi=self.config.visualization.plot_dpi, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Cora dataset experiments")
    parser.add_argument("--config", type=str, default="experiments/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Override output directory")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
        
        # Run experiments
        runner = CoraExperimentRunner(args.config)
        
        # Override output directory if specified
        if args.output_dir:
            runner.config.experiment.output_dir = args.output_dir
        
        results = runner.run_experiments()
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        
        if 'node_classification' in results:
            print("\nNode Classification Results:")
            for metric, stats in results['node_classification'].items():
                if isinstance(stats, dict) and 'mean' in stats:
                    print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        if 'node_reconstruction' in results:
            print("\nNode Reconstruction Results:")
            for metric, stats in results['node_reconstruction'].items():
                if isinstance(stats, dict) and 'mean' in stats:
                    print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        print(f"\nResults saved to: {runner.config.experiment.output_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()