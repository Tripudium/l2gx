"""
Patch Creation and Management for Local2GlobalX

This module implements the core patch-based graph decomposition.
It provides functionality for:

1. **Patch Creation**: Dividing large graphs into overlapping subgraphs (patches)
2. **Overlap Management**: Creating controlled overlap between patches for alignment
3. **Memory Optimization**: Supporting different coordinate storage strategies
4. **Graph Sparsification**: Reducing patch complexity while preserving structure

Key Classes:
- `Patch`: Base class representing a subgraph with embedded coordinates
- `MeanAggregatorPatch`: Aggregates coordinates from multiple patches
- `FilePatch`: Loads patch coordinates from files with transformations

Key Functions:
- `create_patches()`: Main entry point for patch generation
- `create_patch_data()`: Creates patch graph with patches and overlaps as attributes

Example:
    ```python
    from l2gx.patch import create_patches
    from l2gx.graphs import TGraph

    # Create patches from a graph
    patch_graph = create_patches(
        graph=graph,
        num_patches=10,
        min_overlap=32,
        target_overlap=64
    )

    # Access patches and overlaps
    patches = patch_graph.patches
    overlaps = patch_graph.overlap_nodes
    ```
"""

from random import choice
from math import ceil
import copy
from typing import Literal, Optional, Union
from collections.abc import Iterable
from collections import defaultdict
import torch
import numpy as np
import numba
import pymetis

from l2gx.patch.clustering import (
    Partition,
    get_clustering_algorithm,
    CLUSTERING_ALGORITHMS,
)
from l2gx.graphs.tgraph import TGraph
from l2gx.graphs.npgraph import NPGraph, JitGraph
from l2gx.patch.sparsify import (
    resistance_sparsify,
    relaxed_spanning_tree,
    edge_sampling_sparsify,
    hierarchical_sparsify,
    nearest_neighbor_sparsify,
    conductance_weighted_graph,
)

try:
    from l2gx.patch.clustering import fennel_clustering_rust

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


class Patch:
    """
    Patch
    -----
    Represents a local subgraph (patch) of a larger graph, along with its embedding coordinates.

    A Patch object contains:
        - nodes: The list or array of node indices included in this patch.
        - index: A mapping from global node indices to their local index within the patch.
        - coordinates: The embedding coordinates for each node in the patch (can be a numpy array or a lazy loader).

    Typical usage:
        - Constructed from a list of node indices and (optionally) their coordinates.
        - Used as the basic unit for local embedding and for alignment in global embedding algorithms.

    Attributes:
        nodes (np.ndarray): Array of node indices in the patch.
        index (dict): Mapping from node index to coordinate row.
        coordinates (np.ndarray or BaseLazyCoordinates): Embedding coordinates for patch nodes.

    Methods:
        shape: Returns the shape of the coordinates array.
        get_coordinates(nodes): Returns coordinates for a list of node indices.
        get_coordinate(node): Returns the coordinate for a single node.

    Notes:
        - Coordinates can be loaded lazily for large-scale problems.
        - Used throughout the l2gx pipeline for patch-based embedding and alignment.
    """

    def __init__(self, nodes, coordinates=None):
        """
        Initialise a patch from a list of nodes and corresponding coordinates

        Args:
            nodes: Iterable of integer node indices for patch
            coordinates: filename for coordinate file to be loaded on demand
        """
        self.nodes = np.asanyarray(nodes)
        self.index = {int(n): i for i, n in enumerate(nodes)}
        # Example: nodes = [1, 3, 4], index = {1: 0, 3: 1, 4: 2}
        if coordinates is not None:
            self.coordinates = coordinates

    @property
    def shape(self):
        """
        shape of patch coordinates

        (`shape[0]` is the number of nodes in the patch
        and `shape[1]` is the embedding dimension)
        """
        return self.coordinates.shape

    def get_coordinates(self, nodes):
        """
        get coordinates for a list of nodes

        Args:
            nodes: Iterable of node indices
        """
        if isinstance(nodes, int):
            nodes = [nodes]
        return self.coordinates[[self.index[node] for node in nodes], :]

    def get_coordinate(self, node):
        """
        get coordinate for a single node

        Args:
            node: Integer node index
        """
        return self.coordinates[self.index[node], :]

    def __copy__(self):
        """return a copy of the patch"""
        instance = type(self).__new__(type(self))
        instance.__dict__.update(self.__dict__)
        instance.coordinates = copy.copy(self.coordinates)
        return instance


@numba.njit
def geodesic_expand_overlap(
    subgraph, seed_mask, min_overlap, target_overlap, reseed_samples=10
):
    """Expand patch using geodesic distance

    Args:
        subgraph: graph containing patch nodes and all target nodes for potential expansion
        seed_mask: boolean mask indicating seed nodes
        min_overlap: minimum overlap before stopping expansion
        target_overlap: maximum overlap
            (if expansion step results in more overlap, the nodes added are sampled at random)
        reseed_samples: number of random samples when reseeding, default is 10

    Returns:
        index tensor of new nodes to add to patch
    """
    if subgraph.num_nodes < min_overlap:
        raise RuntimeError("Minimum overlap > number of nodes")
    mask = ~seed_mask
    new_nodes = np.flatnonzero(seed_mask)
    overlap = new_nodes
    if overlap.size > target_overlap:
        overlap = np.random.choice(overlap, target_overlap, replace=False)
    while overlap.size < min_overlap:
        new_nodes = subgraph.neighbours(new_nodes)
        new_nodes = new_nodes[mask[new_nodes]]
        if not new_nodes.size:
            # no more connected nodes to add so add some remaining nodes by random sampling
            new_nodes = np.flatnonzero(mask)
            if new_nodes.size > reseed_samples:
                new_nodes = np.random.choice(new_nodes, reseed_samples, replace=False)
        if overlap.size + new_nodes.size > target_overlap:
            new_nodes = np.random.choice(
                new_nodes, target_overlap - overlap.size, replace=False
            )
        if not new_nodes.size:
            raise RuntimeError("Could not reach minimum overlap.")
        mask[new_nodes] = False
        overlap = np.concatenate((overlap, new_nodes))
    return overlap


def merge_small_clusters(
    graph: TGraph, partition_tensor: torch.LongTensor, min_size: int
) -> torch.LongTensor:
    """Merge small clusters with adjacent clusters
    such that all clusters satisfy a minimum size constraint.

    This function iteratively merges the smallest cluster with its neighbouring cluster with the
    maximum normalized cut.

    Args:
        graph: Input graph
        partition_tensor: input partition vector mapping nodes to clusters
        min_size: desired minimum size of clusters

    Returns:
        new partition tensor where small clusters are merged.
    """
    parts = [
        torch.as_tensor(p, device=graph.device) for p in Partition(partition_tensor)
    ]
    num_parts = len(parts)
    part_degs = torch.tensor(
        [graph.degree[p].sum() for p in parts], device=graph.device
    )
    sizes = torch.tensor([len(p) for p in parts], dtype=torch.long)
    smallest_id = torch.argmin(sizes)
    while sizes[smallest_id] < min_size:
        out_neighbour_fraction = torch.zeros(num_parts, device=graph.device)
        p = parts[smallest_id]
        for node in p:
            other = partition_tensor[graph.adj(node)]
            out_neighbour_fraction.scatter_add_(
                0, other, torch.ones(1, device=graph.device).expand(other.shape)
            )
        if out_neighbour_fraction.sum() == 0:
            merge = torch.argsort(sizes)[1]
        else:
            out_neighbour_fraction /= (
                part_degs  # encourage merging with smaller clusters
            )
            out_neighbour_fraction[smallest_id] = 0
            merge = torch.argmax(out_neighbour_fraction)
        if merge > smallest_id:
            new_id = smallest_id
            other = merge
        else:
            new_id = merge
            other = smallest_id

        partition_tensor[parts[other]] = new_id
        sizes[new_id] += sizes[other]
        part_degs[new_id] += part_degs[other]
        parts[new_id] = torch.cat((parts[new_id], parts[other]))
        if other < num_parts - 1:
            partition_tensor[parts[-1]] = other
            sizes[other] = sizes[-1]
            part_degs[other] = part_degs[-1]
            parts[other] = parts[-1]
        num_parts = num_parts - 1
        sizes = sizes[:num_parts]
        part_degs = part_degs[:num_parts]
        parts = parts[:num_parts]
        smallest_id = torch.argmin(sizes)
    return partition_tensor


def create_overlapping_patches(
    graph: TGraph,
    partition_tensor: torch.Tensor,
    patch_graph: TGraph,
    min_overlap: int,
    target_overlap: int,
) -> list[np.ndarray]:
    """Create overlapping patches from a hard partition of an input graph

    Args:
        graph: Input graph
        partition_tensor: partition of input graph
        patch_graph: graph where nodes are clusters of partition
            and an edge indicates that the corresponding patches
            in the output should have at least ``min_overlap`` nodes in common
        min_overlap: minimum overlap for connected patches
        target_overlap: maximum overlap during expansion
            for an edge (additional overlap may result from expansion of other edges)

    Returns:
        list of node-index arrays for patches
    """
    if isinstance(partition_tensor, torch.Tensor):
        partition_tensor = partition_tensor.cpu()

    # TODO: fix protected-access
    # pylint: disable=protected-access
    graph = graph.to(NPGraph)._jitgraph
    patch_graph = patch_graph.to(NPGraph)._jitgraph
    # pylint: enable=protected-access

    parts = Partition(partition_tensor)
    partition_tensor_numpy = partition_tensor.numpy()
    patches = [np.asanyarray(p) for p in parts]
    print("Computing patch overlaps...")
    for i in range(patch_graph.num_nodes):
        part_i = parts[i].numpy()
        part_i.sort()
        patches = _patch_overlaps(
            i,
            part_i,
            partition_tensor_numpy,
            patches,
            graph,
            patch_graph,
            ceil(min_overlap / 2),
            int(target_overlap / 2),
        )

    return patches


@numba.njit
def _patch_overlaps(
    i, part, partition, patches, graph, patch_graph, min_overlap, target_overlap
):
    """Internal function to compute patch overlaps"""
    max_edges = graph.degree[part].sum()
    edge_index = np.empty((2, max_edges), dtype=np.int64)
    adj_index = np.zeros((len(part) + 1,), dtype=np.int64)
    part_index = np.full((graph.num_nodes,), -1, dtype=np.int64)
    part_index[part] = np.arange(len(part))

    patch_index = np.full((patch_graph.num_nodes,), -1, dtype=np.int64)
    patch_index[patch_graph.adj(i)] = np.arange(patch_graph.degree[i])
    source_mask = np.zeros(
        (part.size, patch_graph.degree[i]), dtype=np.bool_
    )  # track source nodes for different patches
    edge_count = 0
    for index, p in enumerate(part):
        targets = graph.adj(p)
        for t in part_index[targets]:
            if t >= 0:
                edge_index[0, edge_count] = index
                edge_index[1, edge_count] = t
                edge_count += 1
        adj_index[index + 1] = edge_count
        pi = patch_index[partition[targets]]
        pi = pi[pi >= 0]
        source_mask[index][pi] = True
    edge_index = edge_index[:, :edge_count]
    subgraph = JitGraph(edge_index, len(part), adj_index, None)

    for it, j in enumerate(patch_graph.adj(i)):
        patches[j] = np.concatenate(
            (
                patches[j],
                part[
                    geodesic_expand_overlap(
                        subgraph,
                        seed_mask=source_mask[:, it],
                        min_overlap=min_overlap,
                        target_overlap=target_overlap,
                    )
                ],
            )
        )
    return patches


def prepare_partition(
    graph: TGraph,
    partition_tensor: Union[torch.LongTensor, list[torch.LongTensor]],
    min_patch_size: Optional[int] = None,
    min_overlap: int = 32,
) -> torch.LongTensor:
    """Prepare partition by merging small clusters if needed

    Args:
        graph: Input graph
        partition_tensor: Initial partition or list of hierarchical partitions
        min_patch_size: Minimum patch size (default: min_overlap)
        min_overlap: Minimum overlap size

    Returns:
        Processed partition tensor
    """
    if min_patch_size is None:
        min_patch_size = min_overlap

    if isinstance(partition_tensor, list):
        return partition_tensor[0]
    else:
        return merge_small_clusters(graph, partition_tensor, min_patch_size)


def create_patch_connectivity(
    graph: TGraph,
    partition_tensor: torch.LongTensor,
    use_conductance_weighting: bool = True,
) -> TGraph:
    """Create patch connectivity graph from partition

    Args:
        graph: Original graph
        partition_tensor: Partition assignment
        use_conductance_weighting: Apply conductance weighting

    Returns:
        Patch connectivity graph
    """
    pg = graph.partition_graph(partition_tensor, self_loops=False).to(TGraph)

    # Connect components if needed
    components = pg.connected_component_ids()
    num_components = components.max() + 1
    if num_components > 1:
        # connect all components
        edges = torch.empty(
            (2, num_components * (num_components - 1) / 2), dtype=torch.long
        )
        comp_lists = [[] for _ in range(num_components)]
        for i, c in enumerate(components):
            comp_lists[c].append(i)
        i = 0
        for c1 in range(num_components):
            for c2 in range(c1 + 1, num_components):
                p1 = choice(comp_lists[c1])
                p2 = choice(comp_lists[c2])
                edges[:, i] = torch.tensor((p1, p2), dtype=torch.long)
                i += 1

        edge_index = torch.cat((pg.edge_index, edges, edges[::-1, :]))
        weights = torch.cat(
            (pg.edge_attr, torch.ones(2 * edges.shape[1], dtype=torch.long))
        )
        pg = TGraph(
            edge_index=edge_index,
            edge_attr=weights,
            ensure_sorted=True,
            num_nodes=pg.num_nodes,
            undir=pg.undir,
        )

    # Apply conductance weighting if requested
    if use_conductance_weighting:
        pg = conductance_weighted_graph(pg)

    return pg


def apply_sparsification(
    patch_graph: TGraph,
    method: Literal["resistance", "rmst", "none", "sample", "neighbors"] = "resistance",
    target_patch_degree: int = 4,
    gamma: int = 0,
    partition_tensor: Optional[Union[torch.LongTensor, list[torch.LongTensor]]] = None,
) -> TGraph:
    """Apply sparsification to patch graph

    Args:
        patch_graph: Patch connectivity graph
        method: Sparsification method
        target_patch_degree: Target degree for sparsification
        gamma: Parameter for RMST method
        partition_tensor: Hierarchical partition for hierarchical sparsification

    Returns:
        Sparsified patch graph
    """
    if method == "resistance":
        if isinstance(partition_tensor, list) and len(partition_tensor) > 1:
            return hierarchical_sparsify(
                patch_graph,
                partition_tensor[1:],
                target_patch_degree,
                sparsifier=resistance_sparsify,
            )
        else:
            return resistance_sparsify(
                patch_graph, target_mean_degree=target_patch_degree
            )
    elif method == "rmst":
        return relaxed_spanning_tree(patch_graph, maximise=True, gamma=gamma)
    elif method == "sample":
        if isinstance(partition_tensor, list) and len(partition_tensor) > 1:
            return hierarchical_sparsify(
                patch_graph,
                partition_tensor[1:],
                target_patch_degree,
                sparsifier=edge_sampling_sparsify,
            )
        else:
            return edge_sampling_sparsify(patch_graph, target_patch_degree)
    elif method == "neighbors":
        if isinstance(partition_tensor, list) and len(partition_tensor) > 1:
            return hierarchical_sparsify(
                patch_graph,
                partition_tensor[1:],
                target_patch_degree,
                sparsifier=nearest_neighbor_sparsify,
            )
        else:
            return nearest_neighbor_sparsify(patch_graph, target_patch_degree)
    elif method == "none":
        return patch_graph
    else:
        raise ValueError(
            f"Unknown sparsify method '{method}', "
            f"should be one of 'resistance', 'rmst', 'sample', 'neighbors', or 'none'."
        )


def compute_patch_overlaps(
    patches: list[Patch], verbose: bool = False
) -> dict[tuple[int, int], list[int]]:
    """Compute overlap information between patches

    Args:
        patches: list of patches
        verbose: Print statistics

    Returns:
        Dictionary mapping (patch_i, patch_j) to list of overlapping nodes
    """
    # Build node->patches index
    node_to_patches = defaultdict(list)
    for i, patch in enumerate(patches):
        for node in patch.nodes:
            node_to_patches[int(node)].append(i)

    # Find overlaps between patches
    overlap_dict = {}
    for node, patch_list in node_to_patches.items():
        if len(patch_list) > 1:
            for i in range(len(patch_list)):
                for j in range(i + 1, len(patch_list)):
                    key = (patch_list[i], patch_list[j])
                    if key not in overlap_dict:
                        overlap_dict[key] = []
                    overlap_dict[key].append(node)

    # Create bidirectional overlap dictionary
    overlap_nodes = {}
    for (i, j), nodes in overlap_dict.items():
        overlap_nodes[(i, j)] = nodes
        overlap_nodes[(j, i)] = nodes  # Add reverse direction

    if verbose and overlap_dict:
        print(f"Computed {len(overlap_dict)} patch overlaps")
        overlap_sizes = [len(nodes) for nodes in overlap_dict.values()]
        print(
            f"Overlap sizes: min={min(overlap_sizes)}, max={max(overlap_sizes)}, "
            f"mean={sum(overlap_sizes) / len(overlap_sizes):.1f}"
        )

    return overlap_nodes


def create_two_patches_optimized(
    graph: TGraph,
    min_overlap: Optional[int] = None,
    target_overlap: Optional[int] = None,
    clustering_method: str = "metis",
    clustering_params: Optional[dict[str, any]] = None,
    verbose: bool = True,
) -> TGraph:
    """
    Optimized function for creating exactly 2 balanced patches with overlap.

    This avoids the hanging issue in the general create_patches function when num_patches=2
    by using a specialized approach for bipartition.

    Args:
        graph: Input graph
        min_overlap: Minimum overlap between patches (default: 10% of partition size)
        target_overlap: Target overlap between patches (default: 20% of partition size)
        clustering_method: Method for initial bipartition (default: 'metis')
        clustering_params: Additional parameters for clustering
        verbose: Print progress information

    Returns:
        Patch graph with 2 patches and controlled overlap
    """
    if verbose:
        print("Creating 2 balanced patches using optimized bipartition")

    # Calculate default overlap based on expected partition size
    expected_size = graph.num_nodes // 2
    if min_overlap is None:
        min_overlap = max(1, expected_size // 10)  # 10% of partition size
    if target_overlap is None:
        target_overlap = max(min_overlap, expected_size // 5)  # 20% of partition size

    if verbose:
        print(f"Overlap parameters: min={min_overlap}, target={target_overlap}")

    # Step 1: Create balanced bipartition
    if clustering_method == "metis":
        try:
            # Convert to adjacency list format for METIS
            adjacency = [[] for _ in range(graph.num_nodes)]
            edge_list = graph.edge_index.t().tolist()
            for u, v in edge_list:
                if u != v:  # Skip self-loops
                    adjacency[u].append(v)

            # Run METIS with balance constraint
            clustering_params = clustering_params or {}
            ufactor = clustering_params.get(
                "balance_tolerance", 30
            )  # 3% imbalance by default

            cut_cost, partition = pymetis.part_graph(
                nparts=2,
                adjacency=adjacency,
                options=pymetis.Options(ufactor=ufactor, contig=True),
            )

            partition_tensor = torch.tensor(partition, dtype=torch.long)

            if verbose:
                size0 = (partition_tensor == 0).sum().item()
                size1 = (partition_tensor == 1).sum().item()
                print(
                    f"METIS bipartition: [{size0}, {size1}] nodes, cut cost: {cut_cost}"
                )

        except ImportError:
            if verbose:
                print("PyMETIS not available, falling back to general clustering")
            partition_tensor = run_clustering(
                graph, 2, clustering_method, clustering_params or {}
            )
    else:
        # Use general clustering method
        partition_tensor = run_clustering(
            graph, 2, clustering_method, clustering_params or {}
        )

    # Step 2: Find boundary nodes between partitions
    boundary_nodes = set()
    edge_list = graph.edge_index.t()

    for edge in edge_list:
        u, v = edge[0].item(), edge[1].item()
        if partition_tensor[u] != partition_tensor[v]:
            boundary_nodes.add(u)
            boundary_nodes.add(v)

    if verbose:
        print(f"Found {len(boundary_nodes)} boundary nodes between partitions")

    # Step 3: Expand each partition to create overlap
    partition0_nodes = set(torch.where(partition_tensor == 0)[0].tolist())
    partition1_nodes = set(torch.where(partition_tensor == 1)[0].tolist())

    # Start with boundary nodes in the overlap
    overlap_nodes = boundary_nodes.copy()

    # Expand overlap to reach target size
    if len(overlap_nodes) < target_overlap:
        # BFS from boundary to expand overlap region
        import collections

        # Convert edge list to adjacency for BFS
        adj_dict = collections.defaultdict(list)
        for edge in edge_list:
            u, v = edge[0].item(), edge[1].item()
            adj_dict[u].append(v)
            adj_dict[v].append(u)

        # BFS from boundary nodes
        queue = collections.deque(boundary_nodes)
        visited = set(boundary_nodes)

        while len(overlap_nodes) < target_overlap and queue:
            node = queue.popleft()
            for neighbor in adj_dict[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    overlap_nodes.add(neighbor)
                    queue.append(neighbor)

                    if len(overlap_nodes) >= target_overlap:
                        break

    # Create final patch node lists
    patch0_nodes = sorted(partition0_nodes | overlap_nodes)
    patch1_nodes = sorted(partition1_nodes | overlap_nodes)

    if verbose:
        print(f"Final patch sizes: [{len(patch0_nodes)}, {len(patch1_nodes)}]")
        print(f"Actual overlap: {len(set(patch0_nodes) & set(patch1_nodes))} nodes")

    # Step 4: Create patch objects
    patch0 = Patch(nodes=patch0_nodes)
    patch1 = Patch(nodes=patch1_nodes)
    patches = [patch0, patch1]

    # Step 5: Create simple patch graph (2 nodes, 1 edge)
    # The patch graph has nodes representing patches and edges representing overlaps
    patch_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()

    patch_graph = TGraph(edge_index=patch_edge_index, num_nodes=2)

    # Add attributes
    patch_graph.patches = patches
    patch_graph.partition = partition_tensor
    patch_graph.overlap_nodes = {
        (0, 1): sorted(set(patch0_nodes) & set(patch1_nodes)),
        (1, 0): sorted(set(patch0_nodes) & set(patch1_nodes)),
    }

    # Add partition assignments to original graph
    graph.partition = partition_tensor

    if verbose:
        print("Created patch graph: 2 nodes, 1 edge")
        print("Patch creation complete")

    return patch_graph


def create_patches(
    graph: TGraph,
    # Clustering parameters (optional if partition_tensor provided)
    patch_size: Optional[int] = None,
    num_patches: Optional[int] = None,
    # Partition parameters
    partition_tensor: Optional[torch.LongTensor] = None,
    clustering_method: str = "metis",
    clustering_params: Optional[dict[str, any]] = None,
    # Patch creation parameters
    min_overlap: Optional[int] = None,
    target_overlap: Optional[int] = None,
    min_patch_size: Optional[int] = None,
    # Sparsification parameters
    sparsify_method: Literal[
        "resistance", "rmst", "none", "sample", "neighbors"
    ] = "resistance",
    target_patch_degree: int = 4,
    gamma: int = 0,
    use_conductance_weighting: bool = True,
    verbose: bool = True,
) -> TGraph:
    """Create patches from a graph with optional clustering

    This is the unified function for patch creation. It can either:
    1. Take a pre-computed partition_tensor and create patches from it
    2. Run clustering first if no partition_tensor is provided

    Args:
        graph: Input graph (TGraph)

        # Clustering parameters (ignored if partition_tensor provided):
        partition_tensor: Pre-computed partition (if None, clustering is run)
        patch_size: Target nodes per patch (exclusive with num_patches)
        num_patches: Target number of patches (exclusive with patch_size)
        clustering_method: Clustering algorithm ('fennel', 'louvain', 'metis', 'spread')
        clustering_params: Additional parameters for clustering algorithm

        # Patch creation parameters:
        min_overlap: Minimum overlap between patches (default: 10% of patch size)
        target_overlap: Target overlap between patches (default: 20% of patch size)
        min_patch_size: Minimum size of patches (default: min_overlap)

        # Sparsification parameters:
        sparsify_method: Graph sparsification method
        target_patch_degree: Target degree for patch graph sparsification
        gamma: Parameter for RMST method
        use_conductance_weighting: Apply conductance weighting to patch graph
        verbose: Print progress information

    Returns:
        Patch graph with the following attributes:
        - patches: list of Patch objects
        - overlap_nodes: Dictionary mapping (patch_i, patch_j) to overlapping node lists
        - partition: Array of length num_nodes indicating cluster assignment for each node

    Example:
        ```python
        # Method 1: Let the function handle clustering
        patch_graph = create_patches(
            graph,
            num_patches=10,
            clustering_method='fennel',
            min_overlap=20
        )

        # Access results
        patches = patch_graph.patches
        overlaps = patch_graph.overlap_nodes
        partition = patch_graph.partition  # Node cluster assignments

        # Method 2: Provide your own partition
        my_partition = custom_clustering_function(graph)
        patch_graph = create_patches(
            graph,
            partition_tensor=my_partition,
            min_overlap=20
        )
        ```
    """
    if verbose:
        print(
            f"Creating patches from graph with {graph.num_nodes} nodes, {graph.num_edges} edges"
        )

    # Special case for 2 patches - use optimized balanced bipartition
    if num_patches == 2 and partition_tensor is None:
        if verbose:
            print("Using optimized 2-patch balanced bipartition")
        return create_two_patches_optimized(
            graph=graph,
            min_overlap=min_overlap,
            target_overlap=target_overlap,
            clustering_method=clustering_method,
            clustering_params=clustering_params,
            verbose=verbose,
        )

    # Step 1: Get partition tensor (either provided or via clustering)
    if partition_tensor is None:
        # Need to run clustering
        if patch_size is not None and num_patches is not None:
            raise ValueError("Cannot specify both patch_size and num_patches")

        if patch_size is None and num_patches is None:
            raise ValueError(
                "Must specify either patch_size or num_patches when partition_tensor is not provided"
            )

        # Calculate parameters
        if patch_size is not None:
            num_patches = max(1, graph.num_nodes // patch_size)
            if verbose:
                print(f"Target patch size: {patch_size} → {num_patches} patches")
        else:
            patch_size = max(1, graph.num_nodes // num_patches)
            if verbose:
                print(f"Target patches: {num_patches} → ~{patch_size} nodes per patch")

        # set overlap defaults based on patch size
        if min_overlap is None:
            min_overlap = max(1, patch_size // 10)  # 10% of patch size

        if target_overlap is None:
            target_overlap = max(min_overlap, patch_size // 5)  # 20% of patch size

        # Run clustering
        if verbose:
            print(f"Running {clustering_method} clustering...")

        clustering_params = clustering_params or {}
        clustering_params.setdefault("verbose", verbose)

        partition_tensor = run_clustering(
            graph, num_patches, clustering_method, clustering_params
        )

        if verbose:
            unique_clusters = len(torch.unique(partition_tensor[partition_tensor >= 0]))
            cluster_sizes = torch.bincount(partition_tensor[partition_tensor >= 0])
            print(
                f"Clustering complete: {unique_clusters} clusters, "
                f"sizes: [{cluster_sizes.min()}, {cluster_sizes.max()}]"
            )
    else:
        # Partition provided - estimate patch size for overlap calculation if not specified
        if min_overlap is None or target_overlap is None:
            unique_clusters = len(torch.unique(partition_tensor))
            avg_patch_size = graph.num_nodes // unique_clusters

            if min_overlap is None:
                min_overlap = max(1, avg_patch_size // 10)

            if target_overlap is None:
                target_overlap = max(min_overlap, avg_patch_size // 5)

    if verbose:
        print(f"Overlap parameters: min={min_overlap}, target={target_overlap}")

    # Step 2: Prepare partition (merge small clusters if needed)
    partition_tensor_processed = prepare_partition(
        graph, partition_tensor, min_patch_size, min_overlap
    )

    if verbose:
        print(
            f"Number of patches after processing: {partition_tensor_processed.max().item() + 1}"
        )

    # Step 3: Create patch connectivity graph
    pg = create_patch_connectivity(
        graph, partition_tensor_processed, use_conductance_weighting
    )

    # Step 4: Apply sparsification
    if verbose:
        print(f"Applying {sparsify_method} sparsification...")

    pg = apply_sparsification(
        pg, sparsify_method, target_patch_degree, gamma, partition_tensor
    )

    if verbose:
        print(f"Average patch degree: {pg.num_edges / pg.num_nodes:.2f}")

    # Step 5: Create overlapping patches
    if verbose:
        print("Creating overlapping patches...")

    patch_arrays = create_overlapping_patches(
        graph, partition_tensor_processed, pg, min_overlap, target_overlap
    )

    # Convert to Patch objects
    patches = [Patch(patch_nodes) for patch_nodes in patch_arrays]

    # Step 6: Add patches and overlap information as attributes
    pg.patches = patches
    pg.overlap_nodes = compute_patch_overlaps(patches, verbose)

    # Step 7: Add partition information as attribute
    # Convert partition tensor to numpy array for easier access
    if isinstance(partition_tensor_processed, torch.Tensor):
        pg.partition = partition_tensor_processed.cpu().numpy()
    else:
        pg.partition = np.array(partition_tensor_processed)

    if verbose:
        patch_sizes = [len(patch.nodes) for patch in patches]
        print(f"Patch creation complete: {len(patches)} patches")
        print(
            f"Patch sizes: [{min(patch_sizes)}, {max(patch_sizes)}], "
            f"avg: {np.mean(patch_sizes):.1f}"
        )
        print(f"Partition attribute added: {len(pg.partition)} node assignments")

    return pg


# High-level patch generation functions


def run_clustering(
    graph: TGraph,
    num_clusters: int,
    method: str,
    use_rust: bool = True,
    params: Optional[dict[str, any]] = None,
) -> torch.Tensor:
    """
    Run clustering algorithm on graph

    Args:
        graph: Input graph
        num_clusters: Target number of clusters
        method: Clustering method name
        use_rust: Use Rust implementation if available
        params: Additional clustering parameters

    Returns:
        Cluster assignment tensor
    """
    params = params or {}

    # Handle Rust Fennel specially
    if method == "fennel" and use_rust and RUST_AVAILABLE:
        try:
            return fennel_clustering_rust(graph, num_clusters, **params)
        except Exception as e:
            print(f"Rust Fennel failed ({e}), falling back to Python")
            method = "fennel"  # Fall back to Python

    # Get clustering function
    if method in CLUSTERING_ALGORITHMS:
        clustering_func = CLUSTERING_ALGORITHMS[method]
    else:
        try:
            clustering_func = get_clustering_algorithm(method)
        except ValueError:
            available = list(CLUSTERING_ALGORITHMS.keys())
            raise ValueError(
                f"Unknown clustering method '{method}'. Available: {available}"
            )

    # Run clustering
    if method == "fennel":
        # Convert TGraph to Raphtory format for Fennel
        try:
            raphtory_graph = graph.to_raphtory()
            return clustering_func(raphtory_graph, num_clusters, **params)
        except Exception:
            # Fallback: use safe implementation
            from l2gx.patch.clustering.fennel import fennel_clustering_safe

            edge_index_np = graph.edge_index.cpu().numpy()
            adj_index_np = graph.adj_index.cpu().numpy()
            clusters_np = fennel_clustering_safe(
                edge_index_np, adj_index_np, graph.num_nodes, num_clusters, **params
            )
            return torch.tensor(clusters_np, dtype=torch.long, device=graph.device)

    elif method == "metis":
        # METIS requires TGraph format
        return clustering_func(graph, num_clusters, **params)

    elif method in ["louvain", "spread"]:
        # These require Raphtory format
        raphtory_graph = graph.to_raphtory()
        return clustering_func(raphtory_graph, num_clusters, **params)

    else:
        # Try the function directly
        return clustering_func(graph, num_clusters, **params)


def rolling_window_graph(n_patches: int, w):
    """Generate patch edges for a rolling window

    Args:
        n_patches: Number of patches
        w: window width (patches connected to the w nearest neighbours on either side)
    """
    if not isinstance(w, Iterable):
        w = range(1, w)
    edges = []
    for i in range(n_patches):
        for wi in w:
            j = i - wi
            if j >= 0 and i != j:
                edges.append((i, j))
        for wi in w:
            j = i + wi
            if j < n_patches and i != j:
                edges.append((i, j))
    return TGraph(edge_index=torch.tensor(edges).T, num_nodes=n_patches, undir=True)


# Utility functions


def list_clustering_methods() -> dict[str, str]:
    """
    list available clustering methods with descriptions

    Returns:
        Dictionary mapping method names to descriptions
    """
    descriptions = {
        "fennel": "Single-pass streaming algorithm with load balancing",
        "louvain": "Modularity-based community detection",
        "metis": "Multi-level graph partitioning (optimal edge cuts)",
        "spread": "Degree-based spreading algorithm",
        "hierarchical": "Multi-level clustering with size constraints",
    }

    if RUST_AVAILABLE:
        descriptions["fennel"] += " (Rust accelerated)"

    return descriptions


def estimate_patch_parameters(
    graph: TGraph,
    target_patch_size: Optional[int] = None,
    target_num_patches: Optional[int] = None,
) -> dict[str, int]:
    """
    Estimate reasonable patch generation parameters for a graph

    Args:
        graph: Input graph
        target_patch_size: Desired patch size (optional)
        target_num_patches: Desired number of patches (optional)

    Returns:
        Dictionary with recommended parameters
    """
    if target_patch_size is not None:
        num_patches = max(1, graph.num_nodes // target_patch_size)
        patch_size = target_patch_size
    elif target_num_patches is not None:
        num_patches = target_num_patches
        patch_size = max(1, graph.num_nodes // num_patches)
    else:
        # Auto-estimate based on graph size
        if graph.num_nodes < 1000:
            num_patches = max(2, graph.num_nodes // 100)
        elif graph.num_nodes < 10000:
            num_patches = max(5, graph.num_nodes // 200)
        else:
            num_patches = max(10, graph.num_nodes // 500)

        patch_size = graph.num_nodes // num_patches

    # Estimate overlaps
    min_overlap = max(1, patch_size // 10)
    target_overlap = max(min_overlap, patch_size // 5)

    # Recommend clustering method based on graph size
    if graph.num_nodes > 10000:
        clustering_method = "fennel"  # Good for large graphs
    elif graph.num_nodes > 1000:
        clustering_method = "metis"  # Good balance of quality and speed
    else:
        clustering_method = "louvain"  # Good for small graphs

    return {
        "num_patches": num_patches,
        "patch_size": patch_size,
        "min_overlap": min_overlap,
        "target_overlap": target_overlap,
        "clustering_method": clustering_method,
    }


# Backward compatibility aliases


def create_patch_data(
    graph: TGraph,
    partition_tensor: torch.LongTensor,
    min_overlap: int,
    target_overlap: int,
    min_patch_size: Optional[int] = None,
    sparsify_method: Literal[
        "resistance", "rmst", "none", "sample", "neighbors"
    ] = "resistance",
    target_patch_degree: int = 4,
    gamma: int = 0,
    use_conductance_weighting: bool = True,
    verbose: bool = False,
) -> TGraph:
    """Backward compatibility wrapper for create_patches.

    This function maintains the old create_patch_data API by calling create_patches
    with the provided partition_tensor.

    Deprecated: Use create_patches() instead.
    """
    return create_patches(
        graph=graph,
        partition_tensor=partition_tensor,
        min_overlap=min_overlap,
        target_overlap=target_overlap,
        min_patch_size=min_patch_size,
        sparsify_method=sparsify_method,
        target_patch_degree=target_patch_degree,
        gamma=gamma,
        use_conductance_weighting=use_conductance_weighting,
        verbose=verbose,
    )


def create_patches_by_size(
    graph: TGraph,
    target_patch_size: int,
    size_tolerance: float = 0.2,
    max_iterations: int = 3,
    **kwargs,
) -> TGraph:
    """Create patches with target size, adjusting clustering until size constraints are met

    This is a convenience wrapper around create_patches that iteratively adjusts
    the number of patches to achieve a target patch size.

    Args:
        graph: Input graph
        target_patch_size: Desired patch size
        size_tolerance: Acceptable size deviation (default: 20%)
        max_iterations: Maximum attempts to achieve target size
        **kwargs: Additional arguments passed to create_patches

    Returns:
        Patch graph with patches and overlap_nodes as attributes
    """
    min_size = int(target_patch_size * (1 - size_tolerance))
    max_size = int(target_patch_size * (1 + size_tolerance))

    for iteration in range(max_iterations):
        num_patches = max(1, graph.num_nodes // target_patch_size)

        patch_graph = create_patches(graph, num_patches=num_patches, **kwargs)

        patch_sizes = [len(patch.nodes) for patch in patch_graph.patches]
        avg_size = np.mean(patch_sizes)

        if min_size <= avg_size <= max_size:
            if kwargs.get("verbose", True):
                print(
                    f"Target size achieved in {iteration + 1} iterations: "
                    f"avg={avg_size:.1f} (target={target_patch_size})"
                )
            return patch_graph

        # Adjust target for next iteration
        if avg_size < min_size:
            target_patch_size = int(target_patch_size * 0.9)
        elif avg_size > max_size:
            target_patch_size = int(target_patch_size * 1.1)

        if kwargs.get("verbose", True):
            print(
                f"Iteration {iteration + 1}: avg_size={avg_size:.1f}, "
                f"adjusting target to {target_patch_size}"
            )

    # Return best attempt
    if kwargs.get("verbose", True):
        print(
            f"Could not achieve target size in {max_iterations} iterations, "
            f"returning best attempt (avg={np.mean(patch_sizes):.1f})"
        )

    return patch_graph


# More backward compatibility aliases
generate_patches_by_size = create_patches_by_size
"""Alias for create_patches_by_size - for backward compatibility"""
