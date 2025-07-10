#!/usr/bin/env python3
"""
Embedding Experiment

This module provides the EmbeddingExperiment class that handles:
- Loading datasets and converting to TGraph
- Computing embeddings (patched or whole graph based on num_patches)
- Saving results and configurations
"""

import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Core L2GX imports
from l2gx.datasets import get_dataset
from l2gx.patch.generate import generate_patches
from l2gx.embedding import get_embedding
from l2gx.align import get_aligner
from l2gx.graphs import TGraph

# Suppress warnings
import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


class EmbeddingExperiment:
    """Core embedding experiment class without visualization functionality"""
    
    def __init__(self, config_path: str):
        """Initialize experiment from config file"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.output_dir = Path(self.config['experiment']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store results for later analysis
        self.results = {}
        self.data = None
        self.embedding = None
        self.patches = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['experiment', 'dataset', 'embedding']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in config")
        
        config.setdefault('patches', {'num_patches': 1})  # Default to whole graph
        config.setdefault('alignment', {'method': 'l2g', 'scale': False})
        
        return config
    
    def load_dataset(self) -> TGraph:
        """Load dataset according to config"""
        dataset_config = self.config['dataset']
        dataset_name = dataset_config['name']
        
        print(f" Loading {dataset_name} dataset...")
        
        # Get dataset with optional parameters
        kwargs = {}
        if 'data_root' in dataset_config:
            kwargs['root'] = dataset_config['data_root']
        
        dataset = get_dataset(dataset_name, **kwargs)
        
        try:
            if hasattr(dataset, 'get') and callable(dataset.get):
                pg_data = dataset.get(0)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    pg_data = dataset.to("torch-geometric")
        except Exception:
            pg_data = dataset.to("torch-geometric")
        
        data = TGraph(
            edge_index=pg_data.edge_index,
            x=pg_data.x,
            y=pg_data.y,
            num_nodes=pg_data.num_nodes
        )
        
        if dataset_config.get('normalize_features', False):
            if data.x is not None:
                row_sums = data.x.sum(dim=1, keepdim=True)
                row_sums[row_sums == 0] = 1.0
                data.x = data.x / row_sums
        
        print(f" {dataset_name} loaded: {data.num_nodes} nodes, {data.num_edges} edges, {data.y.max().item() + 1} classes")
        
        self.results['dataset'] = {
            'name': dataset_name,
            'num_nodes': data.num_nodes,
            'num_edges': data.num_edges,
            'num_classes': data.y.max().item() + 1,
            'num_features': data.x.shape[1] if data.x is not None else 0
        }
        
        self.data = data
        return data
    
    def create_embedding_config(self) -> Dict[str, Any]:
        """Create embedding configuration from config file"""
        embed_config = self.config['embedding'].copy()
        
        # Calculate hidden dimension
        embedding_dim = embed_config['embedding_dim']
        hidden_multiplier = embed_config.get('hidden_dim_multiplier', 2)
        embed_config['hidden_dim'] = embedding_dim * hidden_multiplier
        embed_config.pop('hidden_dim_multiplier', None)
        
        return embed_config
    
    def compute_embedding(self, data: TGraph) -> Tuple[np.ndarray, Optional[List]]:
        """
        Compute embedding (patched or whole graph based on num_patches)
        
        Returns:
            tuple: (embedding_array, patches_list_or_None)
        """
        patch_config = self.config['patches']
        num_patches = patch_config.get('num_patches', 1)
        
        if num_patches == 1:
            # Whole graph embedding
            return self._compute_whole_graph_embedding(data), None
        else:
            # Patched embedding with alignment
            return self._compute_patched_embedding(data)
    
    def _compute_whole_graph_embedding(self, data: TGraph) -> np.ndarray:
        """Compute whole graph embedding"""
        print("\\n Computing whole graph embedding...")
        
        pg_data = data.to_tg()
        
        embed_config = self.create_embedding_config()
        embedder = get_embedding(embed_config['method'], **{k: v for k, v in embed_config.items() if k != 'method'})
        
        embedding = embedder.fit_transform(pg_data)
        
        print(f" Whole graph embedding: {embedding.shape}")
        
        self.results['embedding_type'] = 'whole_graph'
        self.results['num_patches'] = 1
        
        return embedding
    
    def _compute_patched_embedding(self, data: TGraph) -> Tuple[np.ndarray, List]:
        """Compute patched embedding with L2G alignment"""
        print("\\n Computing patched embedding...")
        
        patch_config = self.config['patches']
        
        # Generate patches
        patches, _ = generate_patches(
            data,
            num_patches=patch_config['num_patches'],
            clustering_method=patch_config.get('clustering_method', 'metis'),
            min_overlap=patch_config.get('min_overlap', 32),
            target_overlap=patch_config.get('target_overlap', 64),
            sparsify_method=patch_config.get('sparsify_method', 'resistance'),
            target_patch_degree=patch_config.get('target_patch_degree', 4),
            use_conductance_weighting=patch_config.get('use_conductance_weighting', True),
            verbose=patch_config.get('verbose', False)
        )
        
        print(f"Created {len(patches)} patches")
        
        # Create embedder
        embed_config = self.create_embedding_config()
        embedder = get_embedding(embed_config['method'], **{k: v for k, v in embed_config.items() if k != 'method'})
        
        # Embed each patch
        for i, patch in enumerate(patches):
            print(f"Embedding patch {i+1}/{len(patches)}...")
            
            # Extract patch subgraph
            patch_nodes = torch.tensor(patch.nodes, dtype=torch.long)
            patch_tgraph = data.subgraph(patch_nodes, relabel=True)
            patch_data = patch_tgraph.to_tg()
            
            # Embed the patch
            coordinates = embedder.fit_transform(patch_data)
            patch.coordinates = coordinates
        
        # L2G alignment
        align_config = self.config['alignment']
        print(f"Aligning patches with {align_config['method'].upper()}...")
        aligner = get_aligner(align_config['method'])
        aligner.align_patches(patches, scale=align_config.get('scale', False))
        embedding = aligner.get_aligned_embedding()
        
        print(f" Patched embedding: {embedding.shape}")
        
        # Store embedding type and patch info
        self.results['embedding_type'] = 'patched_l2g'
        self.results['num_patches'] = len(patches)
        self.results['patch_stats'] = {
            'avg_patch_size': np.mean([len(p.nodes) for p in patches]),
            'min_patch_size': min(len(p.nodes) for p in patches),
            'max_patch_size': max(len(p.nodes) for p in patches),
        }
        
        return embedding, patches
    
    def save_results(self, embedding: np.ndarray, patches: Optional[List] = None) -> None:
        """Save embeddings and summary statistics"""
        print("\\n Saving results...")
        
        # Save embedding
        np.save(self.output_dir / "embedding.npy", embedding)
        
        # Update results with embedding stats
        self.results['embedding'] = {
            'shape': embedding.shape,
            'mean': float(embedding.mean()),
            'std': float(embedding.std()),
            'min': float(embedding.min()),
            'max': float(embedding.max())
        }
        
        # Save configuration and results
        self.results['config'] = self.config
        self.results['timestamp'] = datetime.now().isoformat()
        
        with open(self.output_dir / "experiment_results.yaml", 'w') as f:
            yaml.dump(self.results, f, default_flow_style=False)
        
        # Save human-readable summary
        with open(self.output_dir / "summary.txt", 'w') as f:
            f.write(f"{self.config['experiment']['name']}\\n")
            f.write("=" * 50 + "\\n\\n")
            
            f.write(f"Dataset: {self.results['dataset']['name']}\\n")
            f.write(f"Nodes: {self.results['dataset']['num_nodes']}\\n")
            f.write(f"Edges: {self.results['dataset']['num_edges']}\\n")
            f.write(f"Classes: {self.results['dataset']['num_classes']}\\n\\n")
            
            f.write("Embedding Parameters:\\n")
            embed_config = self.config['embedding']
            f.write(f"  Method: {embed_config['method']}\\n")
            f.write(f"  Embedding dim: {embed_config['embedding_dim']}\\n")
            f.write(f"  Hidden dim: {embed_config['embedding_dim'] * embed_config.get('hidden_dim_multiplier', 2)}\\n")
            f.write(f"  Epochs: {embed_config.get('epochs', 1000)}\\n")
            f.write(f"  Learning rate: {embed_config.get('learning_rate', 0.001)}\\n\\n")
            
            f.write(f"Embedding Type: {self.results['embedding_type']}\\n")
            f.write(f"Number of patches: {self.results['num_patches']}\\n")
            
            if patches is not None:
                f.write("\\nPatch Parameters:\\n")
                patch_config = self.config['patches']
                f.write(f"  Clustering method: {patch_config.get('clustering_method', 'metis')}\\n")
                f.write(f"  Target patch degree: {patch_config.get('target_patch_degree', 4)}\\n")
                f.write(f"  Average patch size: {self.results['patch_stats']['avg_patch_size']:.1f}\\n")
                f.write(f"  Patch size range: {self.results['patch_stats']['min_patch_size']}-{self.results['patch_stats']['max_patch_size']}\\n")
        
        print(f" Results saved to {self.output_dir}/")
    
    def run_experiment(self) -> Tuple[np.ndarray, Optional[List], TGraph]:
        """
        Run the complete embedding experiment
        
        Returns:
            tuple: (embedding, patches_or_None, dataset)
        """
        print(f" STARTING: {self.config['experiment']['name']}")
        print("=" * 80)
        print(f"Configuration: {self.config_path}")
        print(f"Output directory: {self.output_dir}")
        
        data = self.load_dataset()
        embedding, patches = self.compute_embedding(data)
        self.save_results(embedding, patches)
        
        self.embedding = embedding
        self.patches = patches
        
        print("\\n EMBEDDING COMPLETE")
        print(f"Generated embedding: {embedding.shape}")
        print(f"Type: {self.results['embedding_type']}")
        print(f"Results saved to: {self.output_dir}")
        
        return embedding, patches, data


def main():
    """Main function for running embedding experiments"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python run/embedding_experiment.py <config_file>")
        print("Example: python run/embedding_experiment.py configs/cora_example.yaml")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    try:
        experiment = EmbeddingExperiment(config_file)
        embedding, patches, data = experiment.run_experiment()
        print(f"\\nEmbedding shape: {embedding.shape}")
        print(f"Patches: {len(patches) if patches else 'None (whole graph)'}")
    except Exception as e:
        print(f" Error running experiment: {e}")
        raise


if __name__ == "__main__":
    main()