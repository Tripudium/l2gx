"""
Base functions and classes for alignment problems.
"""

import scipy as sp
from copy import copy, deepcopy
import json
from typing import Callable, Any

import numpy as np
from tqdm.auto import tqdm
import networkx as nx
from scipy.spatial import procrustes
import scipy.sparse as ss
from collections import defaultdict

# local imports
from l2gx.patch import Patch
from l2gx.utils import ensure_extension

# Random number generator for synchronization
rg = np.random.default_rng()


def procrustes_error(coordinates1, coordinates2):
    """
    compute the procrustes alignment error between two sets of coordinates

    Args:
        coordinates1: First set of coordinates (array-like)
        coordinates2: Second set of coordinates (array-like)

    Note that the two sets of coordinates need to have the same shape.
    """
    return procrustes(coordinates1, coordinates2)[2]


def local_error(patch: Patch, reference_coordinates):
    """
    compute the euclidean distance between patch coordinate and reference
    coordinate for each node in patch

    Args:
        patch:
        reference_coordinates:

    Returns:
        vector of error values
    """
    return np.linalg.norm(
        reference_coordinates[patch.nodes, :] - patch.coordinates, axis=1
    )


def transform_error(transforms):
    """
    Compute the recovery error based on tracked transformations.

    After recovery, all transformations should be constant across patches
    as we can recover the embedding only up to a global scaling/rotation/translation.
    The error is computed as the mean over transformation elements of the standard deviation over patches.

    Args:
        transforms: list of transforms
    """
    return np.mean(np.std(transforms, axis=0))


def orthogonal_MSE_error(rots1, rots2):
    """
    Compute the MSE between two sets of orthogonal transformations up to a global transformation

    Args:
        rots1: First list of orthogonal matrices
        rots2: Second list of orthogonal matrices

    """
    dim = len(rots1[0])
    rots1 = np.asarray(rots1)
    rots1 = rots1.transpose((0, 2, 1))
    rots2 = np.asarray(rots2)
    combined = np.mean(rots1 @ rots2, axis=0)
    _, s, _ = sp.linalg.svd(combined)
    return 2 * (dim - np.sum(s))

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
        patches (list[Patch]): List of patch objects to align
        n_nodes (int): Total number of nodes across all patches
        n_patches (int): Number of patches
        dim (int): Embedding dimension
        scales (np.ndarray): Scale factors applied to each patch
        rotations (np.ndarray): Rotation matrices for each patch
        shifts (np.ndarray): Translation vectors for each patch
        patch_overlap (dict): Dictionary mapping patch pairs to overlapping nodes
        min_overlap (int): Minimum required overlap between patches
    """
    def __init__(self, verbose=False, min_overlap=None):
        """
        Initialize the alignment problem

        Args:
            verbose (bool): If True, print diagnostic information (default: False)
            min_overlap (int): Minimum required overlap between patches (default: dim + 1)
        """
        self.verbose = verbose
        self.min_overlap = min_overlap
        
        # Patch data
        self.patches = []
        self.n_nodes = 0
        self.n_patches = 0
        self.dim = 0
        
        # Transformations
        self.scales = []
        self.rotations = []
        self.shifts = []
        
        # Patch relationships
        self.patch_overlap = {}
        self.patch_index = []
        self.patch_degrees = []
        
        # Results
        self._aligned_embedding = None

    def _register_patches(self, patches: list[Patch], min_overlap: int | None = None):
        """Register patches and validate input.
        
        Args:
            patches: List of Patch objects to align
            min_overlap: Minimum required overlap between patches
            
        Raises:
            ValueError: If patches are invalid or incompatible
            RuntimeError: If patch graph is not connected
        """
        if not patches:
            raise ValueError("No patches provided")
            
        # Validate patch consistency
        dims = [patch.shape[1] for patch in patches]
        if not all(d == dims[0] for d in dims):
            raise ValueError(f"Inconsistent patch dimensions: {dims}")
            
        self.patches = deepcopy(patches)
        self.n_nodes = max(max(patch.index.keys()) for patch in self.patches) + 1
        self.n_patches = len(self.patches)
        self.dim = self.patches[0].shape[1]
        
        # Set minimum overlap requirement
        if min_overlap is None:
            min_overlap = self.min_overlap if self.min_overlap is not None else self.dim + 1
        self.min_overlap = min_overlap

        self.scales = np.ones(self.n_patches)
        self.rotations = np.tile(np.eye(self.dim), (self.n_patches, 1, 1))
        self.shifts = np.zeros((self.n_patches, self.dim))
        self._aligned_embedding = None

        # create an index for the patch membership of each node
        self.patch_index = [[] for _ in range(self.n_nodes)]
        for i, patch in enumerate(self.patches):
            for node in patch.nodes:
                self.patch_index[node].append(i)

        # find patch overlaps
        self.patch_overlap = defaultdict(list)
        for i, patch in enumerate(self.patches):
            assert patch.index is not None
            for node in patch.index:
                for j in self.patch_index[node]:
                    if i != j:
                        self.patch_overlap[i, j].append(node)

        # remove small overlaps
        keys = list(self.patch_overlap.keys())
        for e in keys:
            if e[0] != e[1]:
                if len(self.patch_overlap[e]) < min_overlap:
                    del self.patch_overlap[e]
            else:
                del self.patch_overlap[e]  # remove spurious self-loops

        # find patch degrees
        self.patch_degrees = [0] * self.n_patches
        for i, j in self.patch_overlap.keys():
            self.patch_degrees[i] += 1

        patch_graph = nx.Graph()
        patch_graph.add_edges_from(self.patch_overlap.keys())
        if nx.number_connected_components(patch_graph) > 1:
            raise RuntimeError("patch graph is not connected")

        if self.verbose:
            print(f"mean patch degree: {np.mean(self.patch_degrees)}")

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
        vec = self._synchronise(scaling_mat, 1)
        vec = vec.flatten()
        vec = np.abs(vec)
        vec /= vec.mean()
        vec = np.clip(
            vec, a_min=1 / max_scale, a_max=max_scale, out=vec
        )  # avoid blow-up
        return vec

    def _synchronise(self, matrix: ss.spmatrix, blocksize=1, symmetric=False):
        dim = matrix.shape[0]
        if symmetric:
            matrix = matrix + ss.eye(
                dim
            )  # shift to ensure matrix is positive semi-definite for buckling mode
            eigs, vecs = ss.linalg.eigsh(
                matrix,
                k=blocksize,
                v0=rg.normal(size=dim),
                which="LM",
                sigma=2,
                mode="buckling",
            )
            # eigsh unreliable with multiple (clustered) eigenvalues, only buckling mode seems to help reliably

        else:
            # scaling is not symmetric but Perron-Frobenius applies
            eigs, vecs = ss.linalg.eigs(matrix, k=blocksize, v0=rg.normal(size=dim))
            eigs = eigs.real
            vecs = vecs.real

        order = np.argsort(eigs)
        vecs = vecs[:, order[-1 : -blocksize - 1 : -1]]
        if self.verbose:
            print(f"eigenvalues: {eigs}")
        vecs.shape = (dim // blocksize, blocksize, blocksize)
        return vecs

    def _transform_matrix(
        self,
        transform: Callable[[np.ndarray, np.ndarray], Any],
        dim,
        symmetric_weights=False
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

    def align_patches(self, patches: list[Patch], min_overlap: int | None = None, scale: bool = False) -> 'AlignmentProblem':
        """
        Align the patches to minimize differences on overlapping nodes.
        
        This is the main method that implements the alignment strategy.
        Subclasses must implement this method to provide specific alignment
        algorithms (e.g., Local2Global, geometric neural networks).
        
        Args:
            patches: List of Patch objects to align
            min_overlap: Minimum required overlap between patches
            scale: Whether to perform scale synchronization
            
        Returns:
            Self for method chaining
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
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

