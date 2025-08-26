"""
Patched Embedding Implementation for L2GX.

This module implements the PatchedEmbedding class that performs the complete
Local2Global pipeline: patch decomposition, local embedding, and alignment.
"""


import networkx as nx
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data

from ..align import get_aligner
from ..graphs.tgraph import TGraph
from ..patch import create_patches
from .base import GraphEmbedding
from .registry import get_embedding, register_embedding


@register_embedding("patched", aliases=["patch", "l2g_pipeline"])
class PatchedEmbedding(GraphEmbedding):
    """
    Patched embedding method that implements the complete Local2Global pipeline.

    This embedding method:
    1. Decomposes the input graph into overlapping patches
    2. Embeds each patch individually using a specified embedding method
    3. Aligns the patch embeddings using a specified alignment method
    4. Returns the globally aligned node embeddings

    Example:
        ```python
        embedder = PatchedEmbedding(
            embedding_dim=128,
            embedding_method='vgae',
            num_patches=10,
            clustering_method='metis',
            alignment_method='l2g',  # Currently only l2g is supported
            epochs=500
        )
        embeddings = embedder.fit_transform(graph)
        ```
    """

    def __init__(
        self,
        embedding_dim: int,
        embedding_method: str = "vgae",
        num_patches: int = 10,
        clustering_method: str = "metis",
        alignment_method: str = "l2g",
        min_overlap: int = 10,
        target_overlap: int = 20,
        sparsify_method: str = "resistance",
        target_patch_degree: int = 4,
        enable_scaling: bool = False,
        verbose: bool = False,
        **embedding_kwargs,
    ):
        """
        Initialize the PatchedEmbedding method.

        Args:
            embedding_dim: Target dimensionality of node embeddings
            embedding_method: Name of the embedding method to use for patches
            num_patches: Number of patches to create
            clustering_method: Method for graph clustering ('metis', 'fennel', 'louvain')
            alignment_method: Method for patch alignment (currently only 'l2g' is supported)
            min_overlap: Minimum overlap between connected patches
            target_overlap: Target overlap during patch expansion
            sparsify_method: Method for sparsifying patch graph ('resistance', 'rmst', 'none')
            target_patch_degree: Target degree for patch graph sparsification (default: 4)
            enable_scaling: Whether to enable scale synchronization (can be numerically unstable)
            verbose: Whether to print progress information
            **embedding_kwargs: Additional arguments passed to the embedding method
        """
        super().__init__(embedding_dim, **embedding_kwargs)

        # Validate alignment method
        if alignment_method != "l2g":
            raise ValueError(
                f"Only 'l2g' alignment method is currently supported, got '{alignment_method}'"
            )

        self.embedding_method = embedding_method
        self.num_patches = num_patches
        self.clustering_method = clustering_method
        self.alignment_method = alignment_method
        self.min_overlap = min_overlap
        self.target_overlap = target_overlap
        self.sparsify_method = sparsify_method
        self.target_patch_degree = target_patch_degree
        self.enable_scaling = enable_scaling
        self.verbose = verbose
        self.embedding_kwargs = embedding_kwargs

        # Store fitted components
        self._patches = None
        self._patch_graph = None
        self._aligner = None
        self._tgraph = None

    def fit(
        self, graph: Data | nx.Graph | sp.spmatrix | np.ndarray
    ) -> "PatchedEmbedding":
        """
        Learn patch-based embeddings from the input graph.

        Args:
            graph: Input graph in supported format

        Returns:
            self: Fitted embedding instance
        """
        self._validate_graph_format(graph)

        # Convert to TGraph for processing
        if isinstance(graph, Data):
            self._tgraph = TGraph(
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                x=graph.x,
                y=graph.y,
                num_nodes=graph.num_nodes,
            )
        else:
            # Convert other formats to TGraph via Data
            import torch

            if isinstance(graph, nx.Graph):
                # Convert NetworkX to Data first
                edge_index = np.array(list(graph.edges())).T
                if edge_index.size == 0:
                    edge_index = np.empty((2, 0), dtype=int)
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                data = Data(edge_index=edge_index, num_nodes=graph.number_of_nodes())
            elif isinstance(graph, sp.spmatrix | np.ndarray):
                # Convert sparse/dense matrix to Data
                if isinstance(graph, np.ndarray):
                    graph = sp.csr_matrix(graph)
                edge_index = torch.tensor(np.array(graph.nonzero()), dtype=torch.long)
                data = Data(edge_index=edge_index, num_nodes=graph.shape[0])
            else:
                raise ValueError(f"Unsupported graph type: {type(graph)}")

            self._tgraph = TGraph(
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                x=data.x,
                y=data.y,
                num_nodes=data.num_nodes,
            )

        if self.verbose:
            print(
                f"Processing graph with {self._tgraph.num_nodes} nodes and {self._tgraph.num_edges} edges"
            )

        # Step 1: Generate patches
        if self.verbose:
            print(
                f"Generating {self.num_patches} patches using {self.clustering_method} clustering..."
            )

        self._patch_graph = create_patches(
            self._tgraph,
            num_patches=self.num_patches,
            clustering_method=self.clustering_method,
            min_overlap=self.min_overlap,
            target_overlap=self.target_overlap,
            sparsify_method=self.sparsify_method,
            target_patch_degree=self.target_patch_degree,
            verbose=self.verbose,
        )
        self._patches = self._patch_graph.patches

        if self.verbose:
            patch_sizes = [len(patch.nodes) for patch in self._patches]
            print(
                f"Created {len(self._patches)} patches with sizes: "
                f"min={min(patch_sizes)}, max={max(patch_sizes)}, avg={np.mean(patch_sizes):.1f}"
            )

        # Step 2: Embed each patch
        if self.verbose:
            print(f"Embedding patches using {self.embedding_method}...")

        for i, patch in enumerate(self._patches):
            if self.verbose and i % max(1, len(self._patches) // 10) == 0:
                print(f"  Processing patch {i + 1}/{len(self._patches)}")

            # Extract subgraph for this patch
            import torch

            patch_nodes = torch.tensor(
                patch.nodes, dtype=torch.long, device=self._tgraph.device
            )
            patch_subgraph = self._tgraph.subgraph(
                patch_nodes, relabel=True, keep_x=True, keep_y=True
            )

            # Convert to PyTorch Geometric Data
            patch_data = patch_subgraph.to_tg()

            # Create embedder for this patch
            embedder = get_embedding(
                self.embedding_method,
                embedding_dim=self.embedding_dim,
                **self.embedding_kwargs,
            )

            # Embed the patch
            patch_coordinates = embedder.fit_transform(patch_data)

            # Store coordinates in the patch
            patch.coordinates = patch_coordinates

        # Step 3: set up alignment
        if self.verbose:
            print(f"Setting up {self.alignment_method} alignment...")

        self._aligner = get_aligner(self.alignment_method)

        self.is_fitted = True
        return self

    def transform(
        self, graph: Data | nx.Graph | sp.spmatrix | np.ndarray
    ) -> np.ndarray:
        """
        Generate aligned node embeddings for the given graph.

        Args:
            graph: Input graph (should be the same as used in fit)

        Returns:
            Aligned node embeddings of shape (num_nodes, embedding_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform()")

        # Step 4: Align patches to get global embeddings
        if self.verbose:
            print(f"Aligning patches using {self.alignment_method}...")

        # Align the patches using the selected alignment method
        self._aligner.align_patches(self._patches, scale=self.enable_scaling)

        # Get the aligned embedding
        aligned_coordinates = self._aligner.get_aligned_embedding()

        return aligned_coordinates

    def get_patches(self) -> list | None:
        """
        Get the computed patches after fitting.

        Returns:
            list of Patch objects, or None if not fitted
        """
        return self._patches

    def get_patch_graph(self) -> object | None:
        """
        Get the patch graph after fitting.

        Returns:
            Patch graph object, or None if not fitted
        """
        return self._patch_graph

    def get_params(self) -> dict[str, any]:
        """
        Get parameters for this embedding method.

        Returns:
            Dictionary of parameter names and values
        """
        params = super().get_params()
        params.update(
            {
                "embedding_method": self.embedding_method,
                "num_patches": self.num_patches,
                "clustering_method": self.clustering_method,
                "alignment_method": self.alignment_method,
                "min_overlap": self.min_overlap,
                "target_overlap": self.target_overlap,
                "sparsify_method": self.sparsify_method,
                "target_patch_degree": self.target_patch_degree,
                "enable_scaling": self.enable_scaling,
                "verbose": self.verbose,
            }
        )
        params.update(self.embedding_kwargs)
        return params

    def set_params(self, **params) -> "PatchedEmbedding":
        """
        set parameters for this embedding method.

        Args:
            **params: Parameter names and values to set

        Returns:
            self: For method chaining
        """
        # Handle embedding-specific parameters
        embedding_specific = [
            "embedding_method",
            "num_patches",
            "clustering_method",
            "alignment_method",
            "min_overlap",
            "target_overlap",
            "sparsify_method",
            "target_patch_degree",
            "enable_scaling",
            "verbose",
        ]

        for param in embedding_specific:
            if param in params:
                setattr(self, param, params.pop(param))

        # Handle embedding kwargs
        self.embedding_kwargs.update(params)

        # Call parent set_params for common parameters
        super().set_params(**params)

        return self
