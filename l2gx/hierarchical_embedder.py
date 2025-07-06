"""
Hierarchical Graph Embedding

This module implements a hierarchical approach to graph embedding that balances
the computational complexity of embedding computation with graph decomposition.

The algorithm recursively subdivides large patches until they meet size constraints,
then computes embeddings and aligns them hierarchically.
"""

import torch
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import time

from l2gx.graphs import TGraph
from l2gx.patch import generate_patches, Patch
from l2gx.align.registry import get_aligner


class EmbeddingMethod(ABC):
    """Abstract base class for embedding methods"""
    
    @abstractmethod
    def compute_embedding(self, graph: TGraph) -> torch.Tensor:
        """
        Compute embedding for a graph
        
        Args:
            graph: Input graph
            
        Returns:
            Node embeddings tensor [num_nodes, embed_dim]
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension"""
        pass


class SpectralEmbedding(EmbeddingMethod):
    """Spectral embedding using graph Laplacian eigenvectors"""
    
    def __init__(self, embed_dim: int = 64, normalize: bool = True):
        self.embed_dim = embed_dim
        self.normalize = normalize
    
    def compute_embedding(self, graph: TGraph) -> torch.Tensor:
        """Compute spectral embedding using Laplacian eigenvectors"""
        # Convert to dense adjacency matrix (for small graphs only)
        if graph.num_nodes > 5000:
            raise ValueError(f"SpectralEmbedding not suitable for large graphs ({graph.num_nodes} nodes)")
        
        # Build adjacency matrix
        adj = torch.zeros(graph.num_nodes, graph.num_nodes, device=graph.device)
        adj[graph.edge_index[0], graph.edge_index[1]] = 1.0
        adj = (adj + adj.T) / 2  # Ensure symmetry
        
        # Compute degree matrix
        degree = adj.sum(dim=1)
        deg_sqrt_inv = torch.where(degree > 0, 1.0 / torch.sqrt(degree), 0.0)
        
        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        normalized_adj = deg_sqrt_inv.unsqueeze(1) * adj * deg_sqrt_inv.unsqueeze(0)
        laplacian = torch.eye(graph.num_nodes, device=graph.device) - normalized_adj
        
        # Compute eigenvectors
        eigenvals, eigenvecs = torch.linalg.eigh(laplacian)
        
        # Take smallest eigenvalues (excluding the first one which is ~0)
        embedding = eigenvecs[:, 1:self.embed_dim + 1]
        
        if self.normalize:
            embedding = torch.nn.functional.normalize(embedding, dim=1)
        
        return embedding
    
    @property
    def embedding_dim(self) -> int:
        return self.embed_dim


class RandomWalkEmbedding(EmbeddingMethod):
    """Simple random walk based embedding (placeholder for more sophisticated methods)"""
    
    def __init__(self, embed_dim: int = 64, walk_length: int = 10, num_walks: int = 100):
        self.embed_dim = embed_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
    
    def compute_embedding(self, graph: TGraph) -> torch.Tensor:
        """Compute random walk based embedding"""
        # This is a simplified implementation
        # In practice, you'd use Node2Vec, DeepWalk, or similar
        
        embeddings = torch.randn(graph.num_nodes, self.embed_dim, device=graph.device)
        
        # Simple random walk aggregation
        for _ in range(self.num_walks):
            current_node = torch.randint(0, graph.num_nodes, (1,), device=graph.device).item()
            walk_embedding = torch.zeros(self.embed_dim, device=graph.device)
            
            for step in range(self.walk_length):
                # Add current node's features
                walk_embedding += embeddings[current_node]
                
                # Get neighbors
                neighbors = graph.adj(current_node)
                if len(neighbors) > 0:
                    next_node = neighbors[torch.randint(0, len(neighbors), (1,))].item()
                    current_node = next_node
                else:
                    break
            
            # Update embedding
            embeddings[current_node] = 0.9 * embeddings[current_node] + 0.1 * walk_embedding
        
        return torch.nn.functional.normalize(embeddings, dim=1)
    
    @property
    def embedding_dim(self) -> int:
        return self.embed_dim


class HierarchicalEmbedder:
    """
    Hierarchical graph embedder that recursively decomposes large graphs
    
    This class implements the hierarchical embedding strategy:
    1. Subdivide graph into patches with maximum size constraint
    2. For small patches: compute embedding directly
    3. For large patches: recursively decompose and embed
    4. Align embeddings at each level of the hierarchy
    """
    
    def __init__(
        self,
        embedding_method: EmbeddingMethod,
        alignment_method: str = "l2g",
        max_patch_size: int = 1000,
        max_num_patches: int = 20,
        clustering_method: str = "fennel",
        min_overlap: Optional[int] = None,
        target_overlap: Optional[int] = None,
        alignment_params: Optional[Dict[str, Any]] = None,
        clustering_params: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ):
        """
        Initialize hierarchical embedder
        
        Args:
            embedding_method: Method for computing embeddings on small graphs
            alignment_method: Method for aligning patch embeddings
            max_patch_size: Maximum nodes per patch before recursive decomposition
            max_num_patches: Maximum number of patches at each level
            clustering_method: Clustering algorithm for graph decomposition
            min_overlap: Minimum overlap between patches
            target_overlap: Target overlap between patches
            alignment_params: Parameters for alignment algorithm
            clustering_params: Parameters for clustering algorithm
            verbose: Print progress information
        """
        self.embedding_method = embedding_method
        self.alignment_method = alignment_method
        self.max_patch_size = max_patch_size
        self.max_num_patches = max_num_patches
        self.clustering_method = clustering_method
        self.min_overlap = min_overlap
        self.target_overlap = target_overlap
        self.alignment_params = alignment_params or {}
        self.clustering_params = clustering_params or {}
        self.verbose = verbose
        
        # Statistics tracking
        self.stats = {
            'total_patches_created': 0,
            'max_recursion_depth': 0,
            'embedding_computations': 0,
            'alignment_computations': 0,
            'total_time': 0.0
        }
    
    def embed_graph(self, graph: TGraph) -> torch.Tensor:
        """
        Compute hierarchical embedding for a graph
        
        Args:
            graph: Input graph to embed
            
        Returns:
            Node embeddings tensor [num_nodes, embed_dim]
        """
        start_time = time.time()
        self.stats = {k: 0 if isinstance(v, (int, float)) else v for k, v in self.stats.items()}
        
        if self.verbose:
            print(f"Starting hierarchical embedding for graph with {graph.num_nodes} nodes")
        
        # Compute hierarchical embedding
        embedding = self._embed_recursive(graph, level=0)
        
        # Update statistics
        self.stats['total_time'] = time.time() - start_time
        
        if self.verbose:
            print(f"Hierarchical embedding complete in {self.stats['total_time']:.2f}s")
            print(f"Statistics: {self.stats}")
        
        return embedding
    
    def _embed_recursive(self, graph: TGraph, level: int = 0) -> torch.Tensor:
        """
        Recursively compute embedding for a graph
        
        Args:
            graph: Graph to embed
            level: Current recursion level
            
        Returns:
            Node embeddings
        """
        if self.verbose:
            indent = "  " * level
            print(f"{indent}Level {level}: Processing graph with {graph.num_nodes} nodes")
        
        # Update recursion depth
        self.stats['max_recursion_depth'] = max(self.stats['max_recursion_depth'], level)
        
        # Base case: graph is small enough to embed directly
        if graph.num_nodes <= self.max_patch_size:
            if self.verbose:
                indent = "  " * level
                print(f"{indent}Computing direct embedding ({graph.num_nodes} ≤ {self.max_patch_size})")
            
            self.stats['embedding_computations'] += 1
            return self.embedding_method.compute_embedding(graph)
        
        # Recursive case: decompose into patches
        if self.verbose:
            indent = "  " * level
            print(f"{indent}Decomposing into patches ({graph.num_nodes} > {self.max_patch_size})")
        
        # Generate patches
        patches, patch_graph = generate_patches(
            graph,
            num_patches=min(self.max_num_patches, max(2, graph.num_nodes // self.max_patch_size)),
            clustering_method=self.clustering_method,
            min_overlap=self.min_overlap,
            target_overlap=self.target_overlap,
            clustering_params=self.clustering_params,
            verbose=False  # Suppress patch generation verbosity
        )
        
        self.stats['total_patches_created'] += len(patches)
        
        if self.verbose:
            indent = "  " * level
            patch_sizes = [len(patch.nodes) for patch in patches]
            print(f"{indent}Created {len(patches)} patches, sizes: "
                  f"[{min(patch_sizes)}, {max(patch_sizes)}]")
        
        # Recursively compute embeddings for each patch
        patch_embeddings = []
        patch_graphs = []
        
        for i, patch in enumerate(patches):
            if self.verbose:
                indent = "  " * level
                print(f"{indent}Processing patch {i+1}/{len(patches)} ({len(patch.nodes)} nodes)")
            
            # Create subgraph for this patch
            subgraph = self._extract_subgraph(graph, patch)
            patch_graphs.append(subgraph)
            
            # Recursively compute embedding
            patch_embedding = self._embed_recursive(subgraph, level + 1)
            
            # Update patch coordinates with computed embedding
            patch.coordinates = patch_embedding.cpu().numpy()
            patch_embeddings.append(patch)
        
        # Align patch embeddings
        if self.verbose:
            indent = "  " * level
            print(f"{indent}Aligning {len(patch_embeddings)} patch embeddings")
        
        self.stats['alignment_computations'] += 1
        aligned_embedding = self._align_patches(patch_embeddings)
        
        return aligned_embedding
    
    def _extract_subgraph(self, graph: TGraph, patch: Patch) -> TGraph:
        """
        Extract subgraph corresponding to a patch
        
        Args:
            graph: Original graph
            patch: Patch defining the subgraph
            
        Returns:
            Subgraph as TGraph
        """
        # Get patch nodes
        patch_nodes = torch.tensor(patch.nodes, dtype=torch.long, device=graph.device)
        
        # Create node mapping from original indices to subgraph indices
        node_mapping = {node.item(): i for i, node in enumerate(patch_nodes)}
        
        # Filter edges to only include those within the patch
        edge_mask = torch.isin(graph.edge_index[0], patch_nodes) & torch.isin(graph.edge_index[1], patch_nodes)
        patch_edges = graph.edge_index[:, edge_mask]
        
        # Remap edge indices to subgraph indices
        remapped_edges = torch.zeros_like(patch_edges)
        for i in range(patch_edges.shape[1]):
            remapped_edges[0, i] = node_mapping[patch_edges[0, i].item()]
            remapped_edges[1, i] = node_mapping[patch_edges[1, i].item()]
        
        # Create subgraph
        return TGraph(remapped_edges, num_nodes=len(patch_nodes))
    
    def _align_patches(self, patches: List[Patch]) -> torch.Tensor:
        """
        Align patch embeddings using the specified alignment method
        
        Args:
            patches: List of patches with computed embeddings
            
        Returns:
            Aligned embedding for all nodes
        """
        if len(patches) == 1:
            # No alignment needed for single patch
            return torch.tensor(patches[0].coordinates, dtype=torch.float32)
        
        # Get alignment algorithm
        aligner = get_aligner(self.alignment_method, **self.alignment_params)
        
        # Perform alignment
        aligner.align_patches(patches)
        
        # Return the aligned embedding
        return torch.tensor(aligner.get_aligned_embedding(), dtype=torch.float32)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get embedding computation statistics"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.stats = {
            'total_patches_created': 0,
            'max_recursion_depth': 0,
            'embedding_computations': 0,
            'alignment_computations': 0,
            'total_time': 0.0
        }


class AdaptiveHierarchicalEmbedder(HierarchicalEmbedder):
    """
    Adaptive version that adjusts parameters based on graph characteristics
    """
    
    def __init__(self, embedding_method: EmbeddingMethod, **kwargs):
        super().__init__(embedding_method, **kwargs)
        self.adaptive = True
    
    def _embed_recursive(self, graph: TGraph, level: int = 0) -> torch.Tensor:
        """
        Adaptive recursive embedding with dynamic parameter adjustment
        """
        # Adapt parameters based on graph characteristics
        if level == 0:
            self._adapt_parameters(graph)
        
        return super()._embed_recursive(graph, level)
    
    def _adapt_parameters(self, graph: TGraph):
        """
        Adapt parameters based on graph characteristics
        
        Args:
            graph: Input graph to analyze
        """
        density = graph.num_edges / (graph.num_nodes * (graph.num_nodes - 1) / 2)
        
        # Adjust clustering method based on graph size and density
        if graph.num_nodes > 10000:
            self.clustering_method = "fennel"  # Good for large graphs
        elif density > 0.1:
            self.clustering_method = "louvain"  # Good for dense graphs
        else:
            self.clustering_method = "metis"  # Good for sparse graphs
        
        # Adjust overlap based on density
        if self.min_overlap is None:
            base_overlap = max(1, self.max_patch_size // 20)
            if density > 0.1:
                self.min_overlap = base_overlap * 2
            else:
                self.min_overlap = base_overlap
        
        if self.target_overlap is None:
            self.target_overlap = self.min_overlap * 2
        
        if self.verbose:
            print(f"Adaptive parameters: clustering={self.clustering_method}, "
                  f"overlap=[{self.min_overlap}, {self.target_overlap}], density={density:.4f}")


def create_hierarchical_embedder(
    embedding_method: str = "spectral",
    embed_dim: int = 64,
    alignment_method: str = "l2g",
    max_patch_size: int = 1000,
    adaptive: bool = True,
    **kwargs
) -> HierarchicalEmbedder:
    """
    Factory function to create a hierarchical embedder with common configurations
    
    Args:
        embedding_method: "spectral" or "random_walk"
        embed_dim: Embedding dimension
        alignment_method: Alignment algorithm
        max_patch_size: Maximum patch size before recursion
        adaptive: Use adaptive parameter adjustment
        **kwargs: Additional parameters
        
    Returns:
        Configured HierarchicalEmbedder
    """
    # Create embedding method
    if embedding_method == "spectral":
        embed_method = SpectralEmbedding(embed_dim=embed_dim)
    elif embedding_method == "random_walk":
        embed_method = RandomWalkEmbedding(embed_dim=embed_dim)
    else:
        raise ValueError(f"Unknown embedding method: {embedding_method}")
    
    # Create embedder
    if adaptive:
        return AdaptiveHierarchicalEmbedder(
            embedding_method=embed_method,
            alignment_method=alignment_method,
            max_patch_size=max_patch_size,
            **kwargs
        )
    else:
        return HierarchicalEmbedder(
            embedding_method=embed_method,
            alignment_method=alignment_method,
            max_patch_size=max_patch_size,
            **kwargs
        )


# Example usage functions
def demo_hierarchical_embedding():
    """Demonstrate hierarchical embedding on an example graph"""
    from l2gx.graphs import TGraph
    
    # Create example graph
    num_nodes = 2000
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 5))
    graph = TGraph(edge_index, num_nodes=num_nodes)
    
    print(f"Demo graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Create hierarchical embedder
    embedder = create_hierarchical_embedder(
        embedding_method="spectral",
        embed_dim=32,
        max_patch_size=500,
        adaptive=True,
        verbose=True
    )
    
    # Compute embedding
    embedding = embedder.embed_graph(graph)
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"Statistics: {embedder.get_statistics()}")
    
    return embedding, embedder


if __name__ == "__main__":
    demo_hierarchical_embedding()