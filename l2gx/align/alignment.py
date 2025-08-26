"""
Base functions and classes for alignment problems.
"""

from copy import copy, deepcopy
import json
from typing import Callable
import scipy.sparse as ss
from scipy.sparse.linalg import lsmr
import numpy as np
from tqdm.auto import tqdm
import networkx as nx
from collections import defaultdict
import scipy as sp

# local imports
from l2gx.patch import Patch
from l2gx.graphs.tgraph import TGraph
from l2gx.utils import ensure_extension
from l2gx.align.nla import synchronise

# Random number generator for synchronization
rg = np.random.default_rng()


def _cov_svd(coordinates1: np.ndarray, coordinates2: np.ndarray):
    """
    Compute SVD of covariance matrix between two sets of coordinates

    Args:
        coordinates1: First set of coordinates (array-like)
        coordinates2: Second set of coordinates (array-like)

    Note that the two sets of coordinates need to have the same shape.
    """
    coordinates1 = coordinates1 - coordinates1.mean(axis=0)
    coordinates2 = coordinates2 - coordinates2.mean(axis=0)
    cov = coordinates1.T @ coordinates2
    return sp.linalg.svd(cov)


def relative_orthogonal_transform(coordinates1, coordinates2):
    """
    Find the best orthogonal transformation aligning two sets of coordinates for the same nodes

    Args:
        coordinates1: First set of coordinates (array-like)
        coordinates2: Second set of coordinates (array-like)

    Note that the two sets of coordinates need to have the same shape.
    """
    # Note this is completely equivalent to the approach in
    # "Closed-Form Solution of Absolute Orientation using Orthonormal Matrices"
    # Journal of the Optical Society of America A · July 1988
    U, _, Vh = _cov_svd(coordinates1, coordinates2)
    return U @ Vh


def nearest_orthogonal(mat):
    """
    Compute nearest orthogonal matrix to a given input matrix

    Args:
        mat: input matrix
    """
    U, _, Vh = sp.linalg.svd(mat)
    return U @ Vh


def relative_scale(coordinates1, coordinates2, clamp=1e8):
    """
    compute relative scale of two sets of coordinates for the same nodes

    Args:
        coordinates1: First set of coordinates (array-like)
        coordinates2: Second set of coordinates (array-like)

    Note that the two sets of coordinates need to have the same shape.
    """
    scale1 = np.linalg.norm(coordinates1 - np.mean(coordinates1, axis=0))
    scale2 = np.linalg.norm(coordinates2 - np.mean(coordinates2, axis=0))
    if scale1 > clamp * scale2:
        print("extremely large scale clamped")
        return clamp
    if scale1 * clamp < scale2:
        print("extremely small scale clamped")
        return 1 / clamp
    return scale1 / scale2


class AlignmentProblem:  # pylint: disable=too-many-instance-attributes
    """
    Base class for alignment problems.

    This class provides the foundational framework for aligning embedded patches
    to minimize differences on overlapping nodes. It supports various alignment
    strategies including Local2Global group synchronization and neural network
    based transformations.

    The alignment process consists of:
    1. Patch registration and overlap detection
    2. Scale synchronization (optional)
    3. Rotation/reflection alignment
    4. Translation alignment
    5. Embedding reconstruction

    Attributes:
        patch_graph (TGraph): Patch graph with patches as node features and overlap information
        patches (list[Patch]): list of patch objects to align
        n_nodes (int): Total number of nodes across all patches
        n_patches (int): Number of patches
        dim (int): Embedding dimension
        patch_overlap (dict): Dictionary mapping patch pairs to overlapping nodes
        patch_index (list[list[int]]): list of patch indices for each node
        patch_degrees (list[int]): list of patch degrees
        scales (np.ndarray): Scale factors applied to each patch
        rotations (np.ndarray): Rotation matrices for each patch
        shifts (np.ndarray): Translation vectors for each patch
        _aligned_embedding (np.ndarray): Aligned embedding coordinates
    """

    def __init__(self, verbose=False):
        """
        Initialize the alignment problem

        Args:
            verbose (bool): If True, print diagnostic information (default: False)
        """
        self.verbose = verbose

        # Patch data
        self.patch_graph = None
        self.patches = []
        self.n_nodes = 0
        self.n_patches = 0
        self.dim = 0
        self.min_overlap = 0
        self.patch_overlap = {}  # Example: {(0, 1): [2, 3]}
        self.patch_index = []  # Example: [0, 1, 0, 1], asign nodes to patches
        self.patch_degrees = []  # Example: [2, 2], degree of each patch in patch graph

        # Transformations
        self.scales = []
        self.rotations = []
        self.shifts = []

        # Results
        self._aligned_embedding = None

    def _register_patches(self, patch_graph: TGraph):
        """Register patches using a pre-computed patch graph with overlap information.

        Args:
            patch_graph: Graph encoding patch connectivity and overlaps, with patches as node features

        Raises:
            ValueError: If patch graph doesn't contain patches or they are invalid
            RuntimeError: If patch graph is not connected
        """

        if not hasattr(patch_graph, "patches") or patch_graph.patches is None:
            raise ValueError("Patch graph must contain patches as node features")

        self.patch_graph = patch_graph
        patches = self.patch_graph.patches

        if not patches:
            raise ValueError("No patches found in patch graph")

        # Validate patch graph
        if patch_graph.num_nodes != len(patches):
            raise ValueError(
                f"Patch graph has {patch_graph.num_nodes} nodes but {len(patches)} patches found"
            )

        self.patches = deepcopy(patches)  # not sure if needed
        self.n_nodes = max(max(patch.index.keys()) for patch in self.patches) + 1
        self.n_patches = len(self.patches)
        self.dim = self.patches[0].shape[1]

        self.scales = np.ones(self.n_patches)
        self.rotations = np.tile(np.eye(self.dim), (self.n_patches, 1, 1))
        self.shifts = np.zeros((self.n_patches, self.dim))
        self._aligned_embedding = None

        # Create patch index (maps nodes to patches)
        self.patch_index = [[] for _ in range(self.n_nodes)]
        for i, patch in enumerate(self.patches):
            for node in patch.nodes:
                self.patch_index[node].append(i)

        # Extract overlaps from patch graph
        if (
            hasattr(self.patch_graph, "overlap_nodes")
            and self.patch_graph.overlap_nodes
        ):
            # Use pre-computed overlap nodes directly
            if self.verbose:
                print("Using pre-computed overlap information from patch graph")
            self.patch_overlap = defaultdict(list, self.patch_graph.overlap_nodes)
        else:
            # Compute overlaps from scratch
            if self.verbose:
                print("Computing overlap information from patch edges")

            self.patch_overlap = defaultdict(list)
            processed_pairs = set()
            edge_index = self.patch_graph.edge_index.cpu().numpy()

            for idx in range(edge_index.shape[1]):
                i, j = edge_index[0, idx], edge_index[1, idx]
                if i != j:
                    key = (min(i, j), max(i, j))
                    if key not in processed_pairs:
                        processed_pairs.add(key)
                        # Find overlapping nodes
                        nodes_i = set(self.patches[i].nodes)
                        nodes_j = set(self.patches[j].nodes)
                        overlap = list(nodes_i & nodes_j)
                        if overlap:
                            self.patch_overlap[i, j] = overlap
                            self.patch_overlap[j, i] = overlap
        self.min_overlap = min(len(overlap) for overlap in self.patch_overlap.values())

        # Compute patch degrees
        self.patch_degrees = [0] * self.n_patches
        processed_pairs = set()
        for i, j in self.patch_overlap.keys():
            key = (min(i, j), max(i, j))
            if key not in processed_pairs:
                processed_pairs.add(key)
                self.patch_degrees[i] += 1
                self.patch_degrees[j] += 1

        # Verify connectivity
        if len(self.patch_overlap) > 0:
            # Can do better than using nx.Graph() but this is good enough for now
            patch_graph_nx = nx.Graph()
            unique_edges = set()
            for i, j in self.patch_overlap.keys():
                unique_edges.add((min(i, j), max(i, j)))
            patch_graph_nx.add_edges_from(unique_edges)
            if nx.number_connected_components(patch_graph_nx) > 1:
                raise RuntimeError("Patch graph is not connected")
        elif self.n_patches > 1:
            raise RuntimeError("No overlaps found between patches")

        if self.verbose:
            print(f"Registered {self.n_patches} patches using patch graph")
            print(f"Found {len(processed_pairs)} patch pairs with overlap")
            print(f"Mean patch degree: {np.mean(self.patch_degrees):.2f}")

            # Report overlap statistics
            if len(self.patch_overlap) > 0:
                overlap_sizes = []
                for (i, j), nodes in self.patch_overlap.items():
                    if i < j:  # Count each overlap once
                        overlap_sizes.append(len(nodes))
                if overlap_sizes:
                    print(
                        f"Overlap sizes: min={min(overlap_sizes)}, max={max(overlap_sizes)}, "
                        f"mean={np.mean(overlap_sizes):.1f}"
                    )

    def scale_patches(self, scale_factors=None):
        """
        Synchronise scales of the embeddings for each patch

        Args:
            scale_factors: if provided apply the given scales instead of synchronising
        """
        if scale_factors is None:
            scale_factors = [1 / x for x in self.calc_synchronised_scales()]

        for i, scale in enumerate(scale_factors):
            self.patches[i].coordinates *= scale
            # track transformations
            self.scales[i] *= scale
            self.shifts[i] *= scale
        return self

    def calc_synchronised_scales(self, max_scale=1e8):
        """
        Compute the scaling transformations that best align the patches

        Args:
            max_scale: maximum allowed scale (all scales are clipped to the range [``1/max_scale``, ``max_scale``])
                       (default: 1e8)

        Returns:
            list of scales

        """
        scaling_mat = self._transform_matrix(
            lambda ov1, ov2: relative_scale(ov1, ov2, max_scale), 1
        )
        vec = synchronise(scaling_mat, 1, symmetric=False, method="standard")
        vec = vec.flatten()
        vec = np.abs(vec)
        vec /= vec.mean()
        vec = np.clip(
            vec, a_min=1 / max_scale, a_max=max_scale, out=vec
        )  # avoid blow-up
        return vec

    def translate_patches(self, translations=None):
        """align the patches by translation

        Args:
            translations: If provided, apply the given translations instead of synchronizing

        """
        if translations is None:
            translations = self.calc_synchronised_translations()

        for i, t in enumerate(translations):
            self.patches[i].coordinates += t
            # keep track of transformations
            self.shifts[i] += t
        return self

    def calc_synchronised_translations(self):
        """Compute translations that best align the patches"""
        b = np.empty((len(self.patch_overlap), self.dim))
        row = []
        col = []
        val = []
        for i, ((p1, p2), overlap) in enumerate(self.patch_overlap.items()):
            row.append(i)
            col.append(p1)
            val.append(-1)
            row.append(i)
            col.append(p2)
            val.append(1)
            b[i, :] = np.mean(
                self.patches[p1].get_coordinates(overlap)
                - self.patches[p2].get_coordinates(overlap),
                axis=0,
            )
        A = ss.coo_matrix(
            (val, (row, col)),
            shape=(len(self.patch_overlap), self.n_patches),
            dtype=np.int8,
        )
        A = A.tocsr()
        translations = np.empty((self.n_patches, self.dim))
        for d in range(self.dim):
            translations[:, d] = lsmr(A, b[:, d], atol=1e-16, btol=1e-16)[0]
            # TODO: probably doesn't need to be that accurate, this is for testing
        return translations

    def rotate_patches(
        self, rotations=None, method="standard", sketch_method="gaussian"
    ):
        """align the rotation/reflection of all patches

        Args:
            rotations: If provided, apply the given transformations instead of synchronizing patch rotations
        """
        if rotations is None:
            rotations = (
                rot.T for rot in self.calc_synchronised_rotations(method, sketch_method)
            )

        for i, rot in enumerate(rotations):
            self.patches[i].coordinates = self.patches[i].coordinates @ rot.T
            # track transformations
            self.rotations[i] = self.rotations[i] @ rot.T
            self.shifts[i] = self.shifts[i] @ rot.T
        return self

    def calc_synchronised_rotations(self, method="standard", sketch_method="gaussian"):
        """Compute the orthogonal transformations that best align the patches

        Args:
            method: method to use for synchronization (default: "standard")
        """
        rots = self._transform_matrix(
            relative_orthogonal_transform, self.dim, symmetric_weights=True
        )
        vecs = synchronise(
            rots,
            blocksize=self.dim,
            symmetric=True,
            method=method,
            sketch_method=sketch_method,
        )
        for mat in vecs:
            mat[:] = nearest_orthogonal(mat)
        return vecs

    def _transform_matrix(
        self,
        transform: Callable[[np.ndarray, np.ndarray], any],
        dim,
        symmetric_weights=False,
    ):
        """Calculate matrix of relative transformations between patches

        Args:
            transform: function to compute the relative transformation
            dim: output dimension of transform should be `(dim, dim)`
            symmetric_weights: if true use symmetric weighting (default: False)
        """
        n = self.n_patches  # number of patches
        if dim != 1:
            # construct matrix of rotations as a block-sparse-row matrix
            data = np.empty(shape=(len(self.patch_overlap), dim, dim))
        else:
            data = np.empty(shape=(len(self.patch_overlap),))
        weights = np.zeros(n)
        indptr = np.zeros((n + 1,), dtype=int)
        np.cumsum(self.patch_degrees, out=indptr[1:])
        index = np.empty(shape=(len(self.patch_overlap),), dtype=int)

        keys = sorted(self.patch_overlap.keys())
        # TODO: this could be sped up by a factor of two by not computing rotations twice
        for count, (i, j) in enumerate(keys):
            if i == j:
                element = np.eye(dim)
            else:
                overlap_idxs = self.patch_overlap[i, j]
                # find positions of overlapping nodes in the two reference frames
                overlap1 = self.patches[i].get_coordinates(overlap_idxs)
                overlap2 = self.patches[j].get_coordinates(overlap_idxs)
                element = transform(overlap1, overlap2)
            weight = self.weight(i, j)
            weights[i] += weight
            element *= weight
            data[count] = element
            index[count] = j

        # computed weighted average based on error weights
        if symmetric_weights:
            for i in range(n):
                for ind in range(indptr[i], indptr[i + 1]):
                    data[ind] /= np.sqrt(weights[i] * weights[index[ind]])
        else:
            for i in range(n):
                data[indptr[i] : indptr[i + 1]] /= weights[i]
        if dim == 1:
            matrix = ss.csr_matrix((data, index, indptr), shape=(n, n))
        else:
            matrix = ss.bsr_matrix(
                (data, index, indptr), shape=(dim * n, dim * n), blocksize=(dim, dim)
            )
        return matrix

    def weight(self, i, j):
        """
        Compute the weight for a pair of patches based on their overlap.

        The default implementation uses uniform weighting. Subclasses can
        override this to implement more sophisticated weighting schemes
        based on overlap size, patch quality, or other factors.

        Args:
            i (int): Index of first patch
            j (int): Index of second patch

        Returns:
            float: Weight for the patch pair
        """
        if i == j:
            return 0.0

        # Default: weight by overlap size
        overlap_key = (i, j) if (i, j) in self.patch_overlap else (j, i)
        if overlap_key in self.patch_overlap:
            return len(self.patch_overlap[overlap_key]) / self.min_overlap
        return 0.0

    def mean_embedding(self, out=None):
        """
        Compute node embeddings as the centroid over patch embeddings

        Args:
            out: numpy array to write results to (supply a memmap for large-scale problems that do not fit in ram)
        """
        if out is None:
            embedding = np.zeros((self.n_nodes, self.dim))
        else:
            embedding = out  # important: needs to be zero-initialised

        count = np.array([len(patch_list) for patch_list in self.patch_index])
        for patch in self.patches:
            embedding[patch.nodes] += patch.coordinates

        embedding /= count[:, None]

        return embedding

    def median_embedding(self, out=None):
        if out is None:
            out = np.full((self.n_nodes, self.dim), np.nan)

        for i, pids in tqdm(
            enumerate(self.patch_index),
            total=self.n_nodes,
            desc="Compute median embedding for node",
            disable=self.verbose,
        ):
            if pids:
                points = np.array([self.patches[pid].get_coordinate(i) for pid in pids])
                out[i] = np.median(points, axis=0)
        return out

    def align_patches(
        self, patch_graph: TGraph = None, _scale: bool = False
    ) -> "AlignmentProblem":
        """
        Align the patches to minimize differences on overlapping nodes.

        This is the main method that implements the alignment strategy.
        Subclasses must implement this method to provide specific alignment
        algorithms (e.g., Local2Global, geometric neural networks).

        Args:
            patch_graph: Pre-computed patch graph with patches as node features and overlap information
            scale: Whether to perform scale synchronization
            patches: (Deprecated) list of patches - use patch_graph instead
            min_overlap: (Deprecated) Minimum overlap - use patch_graph instead

        Returns:
            Self for method chaining

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        if patch_graph is not None:
            self._register_patches(patch_graph)
        else:
            raise ValueError("Must provide patch_graph with patches as node features")

        raise NotImplementedError(
            f"align_patches not implemented for {self.__class__.__name__}. "
            "Subclasses must implement this method."
        )

    def get_aligned_embedding(self, scale=False, realign=False, out=None):
        """Return the aligned embedding

        Args:
            scale (bool): If True, rescale patches (default: False)
            realign (bool): If True, recompute aligned embedding even if cached (default: False)
            out (np.ndarray): Optional output array for large-scale problems

        Returns:
            n_nodes x dim numpy array of embedding coordinates

        Raises:
            RuntimeError: If no alignment has been computed yet
        """
        if self._aligned_embedding is None:
            raise RuntimeError(
                "No aligned embedding available. Call align_patches() first."
            )

        if realign or scale:
            # Recompute embedding if requested or scaling needed
            if scale:
                # Apply any additional scaling if needed
                pass  # Implementation depends on specific requirements
            self._aligned_embedding = self.mean_embedding(out=out)

        return self._aligned_embedding

    def save_patches(self, filename):
        """
        save patch embeddings to json file
        Args:
            filename: path to output file


        """
        filename = ensure_extension(filename, ".json")
        patch_dict = {
            str(i): {
                int(node): [float(c) for c in coord]
                for node, coord in zip(patch.index, patch.coordinates)
            }
            for i, patch in enumerate(self.patches)
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(patch_dict, f)

    @classmethod
    def load(cls, filename):
        """
        restore ``AlignmentProblem`` from patch file

        Args:
            filename: path to patch file

        """
        filename = ensure_extension(filename, ".json")
        with open(filename, encoding="utf-8") as f:
            patch_dict = json.load(f)
        patch_list = [None] * len(patch_dict)
        for i, patch_data in patch_dict.items():
            nodes = (int(n) for n in patch_data.keys())
            coordinates = list(patch_data.values())
            patch_list[int(i)] = Patch(nodes, coordinates)
        return cls(patch_list)

    def save_embedding(self, filename):
        """
        save aligned embedding to json file

        Args:
            filename: output filename

        """
        filename = ensure_extension(filename, ".json")
        embedding = {str(i): c for i, c in enumerate(self.get_aligned_embedding())}
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(embedding, f)

    def __copy__(self):
        """return a copy of the alignment problem where all patches are copied."""
        instance = type(self).__new__(type(self))
        for key, value in self.__dict__.items():
            instance.__dict__[key] = copy.copy(value)
        instance.patches = [copy(patch) for patch in self.patches]
        return instance
