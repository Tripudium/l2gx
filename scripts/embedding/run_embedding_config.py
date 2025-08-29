#!/usr/bin/env python3
"""
Configurable Embedding Script

Runs embedding experiments based on YAML configuration files.
Supports L2G, Geo, and hierarchical alignment methods.
"""

import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

from l2gx.align import get_aligner
from l2gx.datasets import get_dataset
from l2gx.embedding import get_embedding
from l2gx.graphs import TGraph

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ConfigurableEmbeddingExperiment:
    """Embedding experiment based on YAML configuration."""

    def __init__(self, config_path: str | Path = "./config/embedding_config_l2g.yaml"):
        """Initialize experiment from config file."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.output_dir = Path(self.config["experiment"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store experiment results
        self.results = {}
        self.data = None
        self.embedding = None
        self.patches = None

    def _load_config(self) -> dict[str, any]:
        """Load and validate configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Check for unified format (new) or legacy format
        is_unified = "patched" in config or "hierarchical" in config
        
        if is_unified:
            # Validate unified format required sections
            required_sections = ["experiment", "dataset"]
            # Must have either patched or hierarchical
            if "patched" not in config and "hierarchical" not in config:
                raise ValueError("Unified config must have either 'patched' or 'hierarchical' section")
            # Must have aligner section for unified format
            if "aligner" not in config:
                raise ValueError("Unified config must have 'aligner' section")
        else:
            # Validate legacy format required sections
            required_sections = ["experiment", "dataset", "embedding", "patches", "alignment"]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in config")

        return config

    def load_dataset(self) -> TGraph:
        """Load dataset according to config."""
        dataset_config = self.config["dataset"]
        dataset_name = dataset_config["name"]

        print(f" Loading {dataset_name} dataset...")

        # Get dataset with optional parameters
        kwargs = {}
        if "data_root" in dataset_config:
            kwargs["root"] = dataset_config["data_root"]

        dataset = get_dataset(dataset_name, **kwargs)
        pg_data = dataset.to("torch-geometric")

        data = TGraph(
            edge_index=pg_data.edge_index,
            x=pg_data.x,
            y=pg_data.y,
            num_nodes=pg_data.num_nodes,
        )

        # Apply feature normalization if requested
        if dataset_config.get("normalize_features", False) and data.x is not None:
            row_sums = data.x.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1.0
            data.x = data.x / row_sums

        print(f" {dataset_name}: {data.num_nodes} nodes, {data.num_edges} edges, {data.y.max().item() + 1} classes")

        self.results["dataset"] = {
            "name": dataset_name,
            "num_nodes": data.num_nodes,
            "num_edges": data.num_edges,
            "num_classes": data.y.max().item() + 1,
            "num_features": data.x.shape[1] if data.x is not None else 0,
        }

        self.data = data
        return data

    def create_embedding_config(self) -> dict[str, any]:
        """Create embedding configuration from config file."""
        embed_config = self.config["embedding"].copy()

        # Calculate hidden dimension if multiplier is provided
        if "hidden_dim_multiplier" in embed_config:
            embedding_dim = embed_config["embedding_dim"]
            hidden_multiplier = embed_config["hidden_dim_multiplier"]
            embed_config["hidden_dim"] = embedding_dim * hidden_multiplier
            embed_config.pop("hidden_dim_multiplier")

        return embed_config

    def compute_embedding(self, data: TGraph) -> tuple[np.ndarray, list | None]:
        """Compute embedding using specified method."""
        # Check if we have a unified configuration
        if "patched" in self.config:
            return self._compute_patched_embedding_unified(data)
        elif "hierarchical" in self.config:
            return self._compute_hierarchical_embedding_unified(data)

        # Legacy patch configuration support
        patch_config = self.config["patches"]

        # Check if hierarchical patching is enabled
        hierarchical_config = patch_config.get("hierarchical", {})
        if hierarchical_config.get("enabled", False):
            return self._compute_hierarchical_embedding(data)
        else:
            # Check if we want whole graph embedding
            num_patches = patch_config.get("num_patches", 1)
            if num_patches == 1:
                return self._compute_whole_graph_embedding(data), None
            else:
                return self._compute_patched_embedding(data)

    def _compute_whole_graph_embedding(self, data: TGraph) -> np.ndarray:
        """Compute whole graph embedding."""
        print("\\n Computing whole graph embedding...")

        pg_data = data.to_tg()
        embed_config = self.create_embedding_config()
        embedder = get_embedding(
            embed_config["method"],
            **{k: v for k, v in embed_config.items() if k != "method"}
        )

        embedding = embedder.fit_transform(pg_data)
        print(f" Whole graph embedding: {embedding.shape}")

        self.results["embedding_type"] = "whole_graph"
        return embedding

    def _compute_patched_embedding(self, data: TGraph) -> tuple[np.ndarray, list]:
        """Compute patched embedding with alignment."""
        print("\\n Computing patched embedding...")

        patch_config = self.config["patches"]

        # Generate patches
        patch_graph = create_patches(
            data,
            num_patches=patch_config["num_patches"],
            clustering_method=patch_config.get("clustering_method", "metis"),
            min_overlap=patch_config.get("min_overlap", 256),
            target_overlap=patch_config.get("target_overlap", 512),
            sparsify_method=patch_config.get("sparsify_method", "resistance"),
            target_patch_degree=patch_config.get("target_patch_degree", 4),
            use_conductance_weighting=patch_config.get("use_conductance_weighting", True),
            verbose=patch_config.get("verbose", False),
        )

        patches = patch_graph.patches
        print(f"Generated {len(patches)} patches")

        # Create embedder
        embed_config = self.create_embedding_config()
        embedder = get_embedding(
            embed_config["method"],
            **{k: v for k, v in embed_config.items() if k != "method"}
        )

        # Embed each patch
        for i, patch in enumerate(patches):
            print(f"  Embedding patch {i + 1}/{len(patches)} ({len(patch.nodes)} nodes)...")

            # Extract patch subgraph
            patch_nodes = torch.tensor(patch.nodes, dtype=torch.long)
            patch_tgraph = data.subgraph(patch_nodes, relabel=True)
            patch_data = patch_tgraph.to_tg()

            # Embed the patch
            coordinates = embedder.fit_transform(patch_data)
            patch.coordinates = coordinates

        # Perform alignment
        embedding = self._align_patches(patches, patch_graph)

        self.results["embedding_type"] = "patched"
        self.results["num_patches"] = len(patches)

        return embedding, patches

    def _compute_patched_embedding_unified(self, data: TGraph) -> tuple[np.ndarray, list | None]:
        """Compute patched embedding using unified framework."""
        print("\\n Computing patched embedding (unified framework)...")

        patched_config = self.config["patched"]
        aligner_config = self.config["aligner"]

        # Create aligner based on configuration
        if aligner_config["method"] == "l2g":
            aligner = get_aligner("l2g")
            if "randomized_method" in aligner_config:
                aligner.randomized_method = aligner_config["randomized_method"]
            if "sketch_method" in aligner_config:
                aligner.sketch_method = aligner_config["sketch_method"]

        elif aligner_config["method"] == "geo":
            geo_kwargs = {
                "method": aligner_config.get("geo_method", "orthogonal"),
                "use_scale": aligner_config.get("use_scale", True),
                "verbose": aligner_config.get("verbose", False)
            }

            if aligner_config.get("use_randomized_init", False):
                geo_kwargs["use_randomized_init"] = True
                geo_kwargs["randomized_method"] = aligner_config.get("randomized_method", "randomized")

            aligner = get_aligner("geo", **geo_kwargs)
        else:
            raise ValueError(f"Unknown alignment method: {aligner_config['method']}")

        # Use unified patched embedding
        embedder = get_embedding(
            "patched",
            embedding_dim=patched_config["embedding_dim"],
            aligner=aligner,
            num_patches=patched_config["num_patches"],
            base_method=patched_config["base_method"],
            clustering_method=patched_config.get("clustering_method", "metis"),
            min_overlap=patched_config.get("min_overlap", 256),
            target_overlap=patched_config.get("target_overlap", 512),
            sparsify_method=patched_config.get("sparsify_method", "resistance"),
            target_patch_degree=patched_config.get("target_patch_degree", 4),
            epochs=patched_config.get("epochs", 100),
            learning_rate=patched_config.get("learning_rate", 0.001),
            patience=patched_config.get("patience", 20),
            hidden_dim=patched_config.get("hidden_dim", patched_config["embedding_dim"] * 2),
            verbose=patched_config.get("verbose", False)
        )

        embedding = embedder.fit_transform(data.to_tg())

        # Get patches for analysis if available
        patches = getattr(embedder, '_patches', None)

        print(f" Unified patched embedding: {embedding.shape}")

        self.results["embedding_type"] = "patched_unified"
        self.results["num_patches"] = patched_config["num_patches"]
        self.results["alignment_method"] = aligner_config["method"]

        return embedding, patches

    def _compute_hierarchical_embedding_unified(self, data: TGraph) -> tuple[np.ndarray, list | None]:
        """Compute hierarchical embedding using unified framework."""
        print("\\n Computing hierarchical embedding (unified framework)...")
        
        hier_config = self.config["hierarchical"]
        aligner_config = self.config["aligner"]
        
        # Create aligner based on configuration
        aligner = get_aligner("l2g")  # Default aligner for hierarchical
        if "randomized_method" in aligner_config:
            aligner.randomized_method = aligner_config["randomized_method"]
        if "sketch_method" in aligner_config:
            aligner.sketch_method = aligner_config["sketch_method"]
        
        # Use unified hierarchical embedding
        embedder = get_embedding(
            "hierarchical",
            embedding_dim=hier_config["embedding_dim"],
            aligner=aligner,
            max_patch_size=hier_config.get("max_patch_size", 800),
            base_method=hier_config.get("base_method", "vgae"),
            min_overlap=hier_config.get("min_overlap", 64),
            target_overlap=hier_config.get("target_overlap", 128),
            epochs=hier_config.get("epochs", 100),
            learning_rate=hier_config.get("learning_rate", 0.001),
            patience=hier_config.get("patience", 20),
            hidden_dim=hier_config.get("hidden_dim", hier_config["embedding_dim"] * 2),
            verbose=hier_config.get("verbose", False)
        )
        
        embedding = embedder.fit_transform(data.to_tg())
        
        # Get patches for analysis if available
        patches = getattr(embedder, '_patches', None)
        
        print(f" Unified hierarchical embedding: {embedding.shape}")
        
        self.results["embedding_type"] = "hierarchical_unified"
        self.results["max_patch_size"] = hier_config.get("max_patch_size", 800)
        self.results["alignment_method"] = "smart_selection"  # Hierarchical uses smart selection
        
        return embedding, patches

    def _compute_hierarchical_embedding(self, data: TGraph) -> tuple[np.ndarray, list]:
        """Compute hierarchical embedding (placeholder - would need hierarchical embedder)."""
        print("\\n Computing hierarchical embedding...")

        # This would require implementing hierarchical embedding logic
        # For now, fall back to regular patched embedding
        print("  Note: Hierarchical embedding not yet implemented, using regular patching")
        return self._compute_patched_embedding(data)

    def _align_patches(self, patches: list, patch_graph=None) -> np.ndarray:
        """Perform patch alignment using specified method."""
        align_config = self.config["alignment"]
        method = align_config["method"]

        print(f"Aligning patches with {method.upper()}...")

        if method == "l2g":
            aligner = get_aligner("l2g")

            # Configure L2G-specific parameters
            if "randomized_method" in align_config:
                aligner.randomized_method = align_config["randomized_method"]
            if "sketch_method" in align_config:
                aligner.sketch_method = align_config["sketch_method"]

            # Perform alignment
            if patch_graph is not None:
                aligner.align_patches(patch_graph)
            else:
                # Create a simple patch graph if needed
                from l2gx.patch.patches import PatchGraph
                pg = PatchGraph(patches)
                aligner.align_patches(pg)

            embedding = aligner.get_aligned_embedding()

        elif method == "geo":
            # Configure Geo-specific parameters
            geo_method = align_config.get("geo_method", "orthogonal")
            num_epochs = align_config.get("num_epochs", 1)
            learning_rate = align_config.get("learning_rate", 0.01)
            use_scale = align_config.get("use_scale", True)

            # Create aligner with supported parameters only
            geo_kwargs = {
                "method": geo_method,
                "use_scale": use_scale,
                "verbose": align_config.get("verbose", False)
            }

            # Add randomized initialization if enabled
            if align_config.get("use_randomized_init", False):
                randomized_method = align_config.get("randomized_method", "randomized")
                geo_kwargs["use_randomized_init"] = True
                geo_kwargs["randomized_method"] = randomized_method
                # Note: sketch_method is not supported by GeoAlignmentProblem constructor

            aligner = get_aligner("geo", **geo_kwargs)

            # Perform alignment
            if patch_graph is not None:
                aligner.align_patches(
                    patch_graph=patch_graph,
                    use_scale=use_scale,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate
                )
            else:
                from l2gx.patch.patches import PatchGraph
                pg = PatchGraph(patches)
                aligner.align_patches(
                    patch_graph=pg,
                    use_scale=use_scale,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate
                )

            embedding = aligner.get_aligned_embedding()

        else:
            raise ValueError(f"Unknown alignment method: {method}")

        print(f" Aligned embedding: {embedding.shape}")
        return embedding

    def save_results(self, embedding: np.ndarray, patches: list | None = None) -> None:
        """Save embeddings and experiment results."""
        print("\\n Saving results...")

        output_config = self.config.get("output", {})
        format_type = output_config.get("format", "npz")

        # Save embedding
        if output_config.get("save_embeddings", True):
            if format_type == "npz":
                output_file = self.output_dir / "embedding_results.npz"
                save_dict = {
                    "embedding": embedding,
                    "labels": self.data.y.cpu().numpy(),
                }
                # Add other results
                for key, value in self.results.items():
                    if key not in ["embedding", "labels"] and not isinstance(value, dict):
                        save_dict[key] = value

                np.savez(output_file, **save_dict)
                print(f" Embeddings saved to {output_file}")

        # Save configuration and metadata
        self.results["config"] = self.config
        self.results["embedding_shape"] = embedding.shape

        metadata_file = self.output_dir / "experiment_metadata.yaml"
        with open(metadata_file, "w") as f:
            yaml.dump(self.results, f, default_flow_style=False)

        print(f" Metadata saved to {metadata_file}")

    def run_experiment(self) -> tuple[np.ndarray, list | None, TGraph]:
        """Run the complete embedding experiment."""
        print(f" STARTING: {self.config['experiment']['name']}")
        print("=" * 80)
        print(f"Configuration: {self.config_path}")
        print(f"Output directory: {self.output_dir}")

        # Run experiment pipeline
        data = self.load_dataset()
        embedding, patches = self.compute_embedding(data)
        self.save_results(embedding, patches)

        self.embedding = embedding
        self.patches = patches

        print("\\n EXPERIMENT COMPLETED")
        print(f"Generated embedding: {embedding.shape}")
        print(f"Results saved to: {self.output_dir}")

        return embedding, patches, data


def main():
    """Main function for running configurable experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run configurable embedding experiment")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--output-name", help="Override output filename")

    args = parser.parse_args()

    try:
        experiment = ConfigurableEmbeddingExperiment(args.config)
        embedding, patches, data = experiment.run_experiment()

        print("\\nExperiment completed successfully!")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Configuration: {args.config}")

    except Exception as e:
        print(f" Error running experiment: {e}")
        raise


if __name__ == "__main__":
    main()
