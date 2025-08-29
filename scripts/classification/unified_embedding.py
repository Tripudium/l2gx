#!/usr/bin/env python3
"""
Unified Configurable Embedding Module

This module provides a single, configurable embedding class that can handle:
- Full graph embeddings
- Patched embeddings with L2G alignment
- Patched embeddings with Geo alignment
- Hierarchical embeddings

The embedding type and parameters are determined entirely by the configuration dictionary.
"""

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Add parent directory to path for L2GX imports
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Add hierarchical directory for binary hierarchical embedding
sys.path.insert(0, str(Path(__file__).parent.parent / "hierarchical"))

from l2gx.align import get_aligner
from l2gx.embedding import get_embedding
from l2gx.graphs import TGraph
from l2gx.patch import create_patches


class UnifiedEmbedding:
    """
    Unified embedding class that handles all embedding types through configuration.
    
    This class provides a single interface for computing embeddings with different
    methods (full graph, L2G, Geo, hierarchical) based entirely on the provided
    configuration dictionary.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the unified embedding with a configuration dictionary.
        
        Args:
            config: Configuration dictionary with the following structure:
                {
                    "type": "full_graph" | "patched" | "hierarchical",
                    "embedding": {
                        "method": "vgae" | "gae" | "svd" | "dgi" | "graphsage",
                        "embedding_dim": int,
                        "hidden_dim": int (optional),
                        "epochs": int,
                        "learning_rate": float,
                        "patience": int,
                        "verbose": bool
                    },
                    "patches": {  # Only for patched type
                        "num_patches": int,
                        "clustering_method": "metis" | "fennel" | "louvain",
                        "min_overlap": int,
                        "target_overlap": int,
                        "sparsify_method": "resistance" | "rmst" | "none",
                        "target_patch_degree": int,
                        "use_conductance_weighting": bool
                    },
                    "alignment": {  # Only for patched type
                        "method": "l2g" | "geo",
                        # For L2G:
                        "randomized_method": "randomized" (optional),
                        "sketch_method": "rademacher" | "gaussian" (optional),
                        # For Geo:
                        "geo_method": "orthogonal" | "euclidean",
                        "num_epochs": int,
                        "learning_rate": float,
                        "use_scale": bool,
                        "use_randomized_init": bool (optional),
                        "verbose": bool
                    },
                    "hierarchical": {  # Only for hierarchical type
                        "max_patch_size": int,
                        "min_overlap": int,
                        "target_overlap": int
                    }
                }
        """
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """Validate the configuration dictionary."""
        if "type" not in self.config:
            raise ValueError("Configuration must specify 'type': 'full_graph', 'patched', or 'hierarchical'")

        if "embedding" not in self.config:
            raise ValueError("Configuration must include 'embedding' section")

        embed_type = self.config["type"]

        if embed_type == "patched":
            if "patches" not in self.config:
                raise ValueError("Patched embedding requires 'patches' configuration")
            if "alignment" not in self.config:
                raise ValueError("Patched embedding requires 'alignment' configuration")
        elif embed_type == "hierarchical":
            if "hierarchical" not in self.config:
                raise ValueError("Hierarchical embedding requires 'hierarchical' configuration")
        elif embed_type != "full_graph":
            raise ValueError(f"Unknown embedding type: {embed_type}")

    def compute_embedding(self, graph: TGraph) -> tuple[np.ndarray, float]:
        """
        Compute embedding based on configuration.
        
        Args:
            graph: Input graph as TGraph
            
        Returns:
            Tuple of (embedding array, computation time in seconds)
        """
        start_time = time.time()

        embed_type = self.config["type"]

        if embed_type == "full_graph":
            embedding = self._compute_full_graph_embedding(graph)
        elif embed_type == "patched":
            embedding = self._compute_patched_embedding(graph)
        elif embed_type == "hierarchical":
            embedding = self._compute_hierarchical_embedding(graph)
        else:
            raise ValueError(f"Unknown embedding type: {embed_type}")

        elapsed_time = time.time() - start_time
        return embedding, elapsed_time

    def _compute_full_graph_embedding(self, graph: TGraph) -> np.ndarray:
        """Compute embedding for the entire graph."""
        embed_config = self.config["embedding"]

        # Convert to PyTorch Geometric format
        pg_data = graph.to_tg()

        # Create embedder
        embedder = get_embedding(
            embed_config["method"],
            embedding_dim=embed_config["embedding_dim"],
            hidden_dim=embed_config.get("hidden_dim", embed_config["embedding_dim"] * 2),
            epochs=embed_config.get("epochs", 100),
            learning_rate=embed_config.get("learning_rate", 0.001),
            patience=embed_config.get("patience", 20),
            verbose=embed_config.get("verbose", False)
        )

        # Compute embedding
        embedding = embedder.fit_transform(pg_data)

        return embedding

    def _compute_patched_embedding(self, graph: TGraph) -> np.ndarray:
        """Compute patched embedding with alignment."""
        embed_config = self.config["embedding"]
        patch_config = self.config["patches"]
        align_config = self.config["alignment"]

        # Create patches
        patch_graph = create_patches(
            graph,
            num_patches=patch_config["num_patches"],
            clustering_method=patch_config.get("clustering_method", "metis"),
            min_overlap=patch_config.get("min_overlap", 256),
            target_overlap=patch_config.get("target_overlap", 512),
            sparsify_method=patch_config.get("sparsify_method", "resistance"),
            target_patch_degree=patch_config.get("target_patch_degree", 4),
            use_conductance_weighting=patch_config.get("use_conductance_weighting", True),
            verbose=patch_config.get("verbose", False)
        )

        patches = patch_graph.patches

        # Create embedder
        embedder = get_embedding(
            embed_config["method"],
            embedding_dim=embed_config["embedding_dim"],
            hidden_dim=embed_config.get("hidden_dim", embed_config["embedding_dim"] * 2),
            epochs=embed_config.get("epochs", 100),
            learning_rate=embed_config.get("learning_rate", 0.001),
            patience=embed_config.get("patience", 20),
            verbose=embed_config.get("verbose", False)
        )

        # Embed each patch
        for patch in patches:
            patch_nodes = torch.tensor(patch.nodes, dtype=torch.long)
            patch_tgraph = graph.subgraph(patch_nodes, relabel=True)
            patch_data = patch_tgraph.to_tg()

            coordinates = embedder.fit_transform(patch_data)
            patch.coordinates = coordinates

        # Perform alignment
        embedding = self._align_patches(patch_graph, align_config)

        return embedding

    def _align_patches(self, patch_graph, align_config: dict[str, Any]) -> np.ndarray:
        """Align patches using specified method."""
        method = align_config["method"]

        if method == "l2g":
            # Create L2G aligner
            aligner = get_aligner("l2g")

            # Configure optional parameters
            if "randomized_method" in align_config:
                aligner.randomized_method = align_config["randomized_method"]
            if "sketch_method" in align_config:
                aligner.sketch_method = align_config["sketch_method"]

            # Perform alignment
            aligner.align_patches(patch_graph)
            embedding = aligner.get_aligned_embedding()

        elif method == "geo":
            # Configure Geo aligner
            geo_kwargs = {
                "method": align_config.get("geo_method", "orthogonal"),
                "use_scale": align_config.get("use_scale", True),
                "verbose": align_config.get("verbose", False)
            }

            # Add randomized initialization if specified
            if align_config.get("use_randomized_init", False):
                geo_kwargs["use_randomized_init"] = True
                geo_kwargs["randomized_method"] = align_config.get("randomized_method", "randomized")
                if "sketch_method" in align_config:
                    geo_kwargs["sketch_method"] = align_config["sketch_method"]

            aligner = get_aligner("geo", **geo_kwargs)

            # Perform alignment
            aligner.align_patches(
                patch_graph=patch_graph,
                use_scale=align_config.get("use_scale", True),
                num_epochs=align_config.get("num_epochs", 1),
                learning_rate=align_config.get("learning_rate", 0.01)
            )

            embedding = aligner.get_aligned_embedding()

        else:
            raise ValueError(f"Unknown alignment method: {method}")

        return embedding

    def _compute_hierarchical_embedding(self, graph: TGraph) -> np.ndarray:
        """Compute hierarchical embedding."""
        embed_config = self.config["embedding"]
        hier_config = self.config["hierarchical"]

        # Import here to avoid circular imports
        from binary_hierarchical_embedding import BinaryHierarchicalEmbedding

        # Create hierarchical embedder
        embedder = BinaryHierarchicalEmbedding(
            max_patch_size=hier_config["max_patch_size"],
            embedding_dim=embed_config["embedding_dim"],
            embedding_method=embed_config["method"],
            min_overlap=hier_config.get("min_overlap", 64),
            target_overlap=hier_config.get("target_overlap", 128),
            epochs=embed_config.get("epochs", 100),
            verbose=embed_config.get("verbose", False)
        )

        # Build tree and embed
        embedder.graph = graph
        embedder.labels = graph.y.numpy() if hasattr(graph, 'y') else None

        embedder.root = embedder.build_hierarchical_tree(graph)
        embedder.embed_leaf_patches()
        embedder.hierarchical_alignment(embedder.root)

        embedding = embedder.root.embedding

        return embedding


def create_embedding_config(method_name: str, method_config: dict[str, Any],
                           embedding_dim: int) -> dict[str, Any]:
    """
    Create a unified embedding configuration from method-specific config.
    
    Args:
        method_name: Name of the method (e.g., "full_graph", "l2g_rademacher")
        method_config: Method-specific configuration from YAML
        embedding_dim: Target embedding dimension
        
    Returns:
        Unified configuration dictionary for UnifiedEmbedding
    """
    config = {"embedding": {}}

    # Extract base embedding configuration
    if "base_embedding" in method_config:
        base_embed = method_config["base_embedding"]
        config["embedding"] = {
            "method": base_embed["method"],
            "embedding_dim": embedding_dim,
            "epochs": base_embed.get("epochs", 100),
            "learning_rate": base_embed.get("learning_rate", 0.001),
            "patience": base_embed.get("patience", 20),
            "verbose": base_embed.get("verbose", False)
        }

    # Determine embedding type and add specific configuration
    if method_name == "full_graph":
        config["type"] = "full_graph"

    elif method_name in ["l2g_rademacher", "l2g_standard", "geo_rademacher"]:
        config["type"] = "patched"

        # Add patch configuration
        if "patches" in method_config:
            config["patches"] = method_config["patches"].copy()

        # Add alignment configuration
        if "alignment" in method_config:
            config["alignment"] = method_config["alignment"].copy()

    elif method_name == "hierarchical_l2g":
        config["type"] = "hierarchical"

        # Add hierarchical configuration
        if "hierarchical" in method_config:
            config["hierarchical"] = method_config["hierarchical"].copy()

    else:
        raise ValueError(f"Unknown method name: {method_name}")

    return config
