#  Copyright (c) 2021. Lucas G. S. Jeub
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import numpy as np
import scipy as sp
import scipy.sparse as ss
from scipy.sparse.linalg import lsmr

from l2gx.patch import Patch
from l2gx.align.registry import register_aligner
from l2gx.align.alignment import AlignmentProblem

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

@register_aligner("l2g")
class L2GAlignmentProblem(AlignmentProblem):
    """
    Implements the standard local2global algorithm using an unweighted patch graph
    """
    def __init__(
        self,
        verbose=False
    ):
        """
        Initialise the alignment problem with a list of patches

        Args:
            verbose(bool): if True print diagnostic information (default: ``False``)

        """
        super().__init__(
            verbose=verbose
        )

    def rotate_patches(self, rotations=None):
        """align the rotation/reflection of all patches

        Args:
            rotations: If provided, apply the given transformations instead of synchronizing patch rotations
        """
        if rotations is None:
            rotations = (rot.T for rot in self.calc_synchronised_rotations())

        for i, rot in enumerate(rotations):
            self.patches[i].coordinates = self.patches[i].coordinates @ rot.T
            # track transformations
            self.rotations[i] = self.rotations[i] @ rot.T
            self.shifts[i] = self.shifts[i] @ rot.T
        return self

    def calc_synchronised_rotations(self):
        """Compute the orthogonal transformations that best align the patches"""
        rots = self._transform_matrix(
            relative_orthogonal_transform, self.dim, symmetric_weights=True
        )
        vecs = self._synchronise(rots, blocksize=self.dim, symmetric=True)
        for mat in vecs:
            mat[:] = nearest_orthogonal(mat)
        return vecs

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

    def align_patches(self, patches: list[Patch], min_overlap: int | None = None, scale: bool = True):
        self._register_patches(patches, min_overlap)
        if scale:
            self.scale_patches()
        self.rotate_patches()
        self.translate_patches()
        self._aligned_embedding = self.mean_embedding()
        return self
