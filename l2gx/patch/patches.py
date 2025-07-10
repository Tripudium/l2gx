# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments

"""
Patch Creation and Management for Local2Global Algorithm

This module implements the core patch-based graph decomposition used in the Local2Global
algorithm. It provides functionality for:

1. **Patch Creation**: Dividing large graphs into overlapping subgraphs (patches)
2. **Overlap Management**: Creating controlled overlap between patches for alignment
3. **Memory Optimization**: Supporting different coordinate storage strategies
4. **Graph Sparsification**: Reducing patch complexity while preserving structure

Key Classes:
- `Patch`: Base class representing a subgraph with embedded coordinates
- `MeanAggregatorPatch`: Aggregates coordinates from multiple patches
- `FilePatch`: Loads patch coordinates from files with transformations

Key Functions:
- `create_patch_data()`: Main entry point for patch creation
- `create_overlapping_patches()`: Creates patches with controlled overlap
- `geodesic_expand_overlap()`: Expands patches using geodesic distance

The patch concept is central to the Local2Global approach:
- **Local Phase**: Each patch gets embedded independently 
- **Global Phase**: Patches are aligned using overlapping regions
- **Scalability**: Enables processing of graphs too large for direct embedding

Example:
    ```python
    from l2gx.patch import create_patch_data
    from l2gx.graphs import TGraph
    
    # Create patches from a graph
    patches = create_patch_data(
        graph=graph,
        partition=clustering_result,
        overlap_size=10,
        sparsify_overlap=True
    )
    ```
"""

from random import choice
from math import ceil
import copy
from typing import Literal, Callable
from collections.abc import Iterable
import torch
import numpy as np
from tqdm.auto import tqdm
import numba

from .clustering import Partition
from ..graphs.tgraph import TGraph
from ..graphs.npgraph import NPGraph, JitGraph
from .sparsify import (
    resistance_sparsify,
    relaxed_spanning_tree,
    edge_sampling_sparsify,
    hierarchical_sparsify,
    nearest_neighbor_sparsify,
    conductance_weighted_graph,
)

from l2gx.patch.lazy import (
    BaseLazyCoordinates,
    LazyMeanAggregatorCoordinates,
    LazyFileCoordinates,
)


class Patch:
    """
    Class for patch embedding
    """

    index = None
    """mapping of node index to patch coordinate index"""

    coordinates = None
    """patch embedding coordinates"""

    def __init__(self, nodes, coordinates=None):
        """
        Initialise a patch from a list of nodes and corresponding coordinates

        Args:
            nodes: Iterable of integer node indices for patch
            coordinates: filename for coordinate file to be loaded on demand
        """
        self.nodes = np.asanyarray(nodes)
        self.index = {int(n): i for i, n in enumerate(nodes)}
        if coordinates is not None:
            if not isinstance(coordinates, BaseLazyCoordinates):
                self.coordinates = np.asanyarray(coordinates)
            else:
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


class MeanAggregatorPatch(Patch):
    """Patch class that aggregates multiple patches by taking their mean coordinates."""

    def __init__(self, patches):
        coordinates = LazyMeanAggregatorCoordinates(patches)
        super().__init__(coordinates.nodes, coordinates)

    @property
    def patches(self):
        return self.coordinates.patches

    def get_coordinate(self, node):
        # avoid double index conversion
        return self.coordinates.get_coordinates([node])

    def get_coordinates(self, nodes):
        # avoid double index conversion
        return self.coordinates.get_coordinates(nodes)


class FilePatch(Patch):
    """Patch class that loads coordinates from a file with optional transformations (shift, scale, rotation)."""

    def __init__(self, nodes, filename, shift=None, scale=None, rot=None):
        super().__init__(
            nodes, LazyFileCoordinates(filename, shift=shift, scale=scale, rot=rot)
        )


@numba.njit
def geodesic_expand_overlap(
    subgraph, seed_mask, min_overlap, target_overlap, reseed_samples=10
):
    """Expand patch

    Args:
        subgraph: graph containing patch nodes and all target nodes for potential expansion

        seed_mask: [description]

        min_overlap: minimum overlap before stopping expansion

        target_overlap: maximum overlap
            (if expansion step results in more overlap, the nodes added are sampled at random)

        reseed_samples: [description] default is 10

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
):
    """Merge small clusters with adjacent clusters
    such that all clusters satisfy a minimum size constraint.

    This function iteratively merges the smallest cluster with its neighbouring cluster with the
    maximum normalized cut.

    Args:
        graph (TGraph): Input graph

        partition_tensor (torch.LongTensor): input partition vector mapping nodes to clusters

        min_size (int): desired minimum size of clusters

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
):
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
        list of node-index tensors for patches

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
    for i in tqdm(range(patch_graph.num_nodes), desc="enlarging patch overlaps"):
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


# TODO: fix too-many-branches
# pylint: disable=too-many-branches
def create_patch_data(
    graph: TGraph,
    partition_tensor: torch.LongTensor,
    min_overlap: int,
    target_overlap: int,
    min_patch_size: int | None = None,
    sparsify_method: Literal["resistance", "rmst", "none", "sample", "neighbors"] = "resistance",
    target_patch_degree: int = 4,
    gamma: int = 0,
    use_conductance_weighting: bool = True,
    verbose: bool = False,
) -> tuple[list, object]:
    """Divide data into overlapping patches

    Args:
        graph (TGraph): input data

        partition_tensor (torch.LongTensor): starting partition for creating patches

        min_overlap: minimum patch overlap for connected patches

        target_overlap: maximum patch overlap during expansion
            of an edge of the patch graph

        min_patch_size: minimum size of patches, defauls is None

        sparsify_method: method for sparsifying patch graph
            (one of ``'resistance'``, ``'rmst'``, ``'none'``), default is ``'resistance'``

        target_patch_degree: target patch degree for
            ``sparsify_method='resistance'``, default is 4

        gamma: ``gamma`` value for use with ``sparsify_method='rmst'``, default is 0

        use_conductance_weighting: if true, apply conductance weighting to patch graph, default is True

        verbose: if true, print some info about created patches, default is False

    Returns:
        list of patch data, patch graph

    """
    if min_patch_size is None:
        min_patch_size = min_overlap

    if isinstance(partition_tensor, list):
        partition_tensor_0 = partition_tensor[0]
    else:
        partition_tensor = merge_small_clusters(graph, partition_tensor, min_patch_size)
        partition_tensor_0 = partition_tensor

    if verbose:
        print(f"number of patches: {partition_tensor_0.max().item() + 1}")
    pg = graph.partition_graph(partition_tensor_0, self_loops=False).to(TGraph)

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
    
    # Apply conductance weighting consistently (like the old implementation)
    if use_conductance_weighting:
        pg = conductance_weighted_graph(pg)

    if sparsify_method == "resistance":
        if isinstance(partition_tensor, list):
            pg = hierarchical_sparsify(
                pg,
                partition_tensor[1:],
                target_patch_degree,
                sparsifier=resistance_sparsify,
            )
        else:
            pg = resistance_sparsify(pg, target_mean_degree=target_patch_degree)
    elif sparsify_method == "rmst":
        pg = relaxed_spanning_tree(pg, maximise=True, gamma=gamma)
    elif sparsify_method == "sample":
        if isinstance(partition_tensor, list):
            pg = hierarchical_sparsify(
                pg,
                partition_tensor[1:],
                target_patch_degree,
                sparsifier=edge_sampling_sparsify,
            )
        else:
            pg = edge_sampling_sparsify(pg, target_patch_degree)
    elif sparsify_method == "neighbors":
        if isinstance(partition_tensor, list):
            pg = hierarchical_sparsify(
                pg,
                partition_tensor[1:],
                target_patch_degree,
                sparsifier=nearest_neighbor_sparsify,
            )
        else:
            pg = nearest_neighbor_sparsify(pg, target_patch_degree)
    elif sparsify_method == "none":
        pass
    else:
        raise ValueError(
            f"Unknown sparsify method '{sparsify_method}', "
            f"should be one of 'resistance', 'rmst', or 'none'."
        )

    if verbose:
        print(f"average patch degree: {pg.num_edges / pg.num_nodes}")

    patch_arrays = create_overlapping_patches(
        graph, partition_tensor_0, pg, min_overlap, target_overlap
    )
    
    # Convert numpy arrays to Patch objects
    patches = [Patch(patch_nodes) for patch_nodes in patch_arrays]
    
    return patches, pg


# pylint: enable=too-many-branches


def create_patch_graph(
    graph: TGraph,
    max_num_patches: int,
    min_overlap: int,
    target_overlap: int,
    clustering_function: Callable[[TGraph, int], torch.Tensor],
) -> tuple[list, TGraph]:
    """Create a patch graph from a graph

    Args:
        graph (TGraph): input graph
        max_num_patches (int): maximum number of patches
        min_overlap (int): minimum overlap for connected patches
        target_overlap (int): maximum overlap during expansion

    Returns:
        list of patch data, patch graph
    """
    partition_tensor = clustering_function(graph, max_num_patches)
    return create_patch_data(graph, partition_tensor, min_overlap, target_overlap)


def rolling_window_graph(n_patches: int, w):
    """Generate patch edges for a rolling window

    Args:
        n_patches (int): Number of patches

        w (): window width (patches connected to the w nearest neighbours on either side)

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
