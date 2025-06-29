"""Utility functions for patch operations.

This module provides utility functions for working with patches in the Local2Global
algorithm, including coordinate transformations, error computations, and file operations.
"""
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
from scipy.spatial import procrustes
from pathlib import Path
from .patches import Patch


def seed(new_seed):
    """Change seed of random number generator.

    Args:
        new_seed: New seed value
    """
    np.random.default_rng(new_seed)


def ensure_extension(filename, extension):
    """Check filename for extension and add it if necessary.

    Args:
        filename: Input filename
        extension: Desired extension (including `.`)

    Returns:
        filename with extension added

    Raises:
        ValueError: if filename has the wrong extension
    """
    filename = Path(filename)
    if filename.suffix == "":
        filename = filename.with_suffix(extension)
    elif filename.suffix != extension:
        raise ValueError(
            f"filename should have extension {extension}, not {filename.suffix}"
        )
    return filename


def random_gen(new_seed=None) -> np.random.Generator:
    """Change seed of random number generator.

    Args:
        new_seed: New seed value, defaults to None

    Returns:
        Random number generator instance
    """
    return np.random.default_rng(new_seed)


# Global random generator instance
rg = random_gen()
eps = np.finfo(float).eps


def procrustes_error(coordinates1: np.ndarray, coordinates2: np.ndarray) -> float:
    """Compute the procrustes alignment error between two sets of coordinates.

    Args:
        coordinates1: First set of coordinates (array-like)
        coordinates2: Second set of coordinates (array-like)

    Returns:
        Procrustes alignment error

    Note:
        The two sets of coordinates need to have the same shape.
    """
    return procrustes(coordinates1, coordinates2)[2]


def local_error(patch: Patch, reference_coordinates) -> np.ndarray:
    """Compute the euclidean distance between patch coordinates and reference coordinates.

    Args:
        patch: Patch object containing nodes and coordinates
        reference_coordinates: Reference coordinate array

    Returns:
        Vector of error values for each node in the patch
    """
    return np.linalg.norm(
        reference_coordinates[patch.nodes, :] - patch.coordinates, axis=1
    )


def transform_error(transforms) -> float:
    """Compute the recovery error based on tracked transformations.

    After recovery, all transformations should be constant across patches as we can
    recover the embedding only up to a global scaling/rotation/translation.

    The error is computed as the mean over transformation elements of the standard
    deviation over patches.

    Args:
        transforms: List of transformation matrices

    Returns:
        Mean standard deviation across transformation elements
    """
    return np.mean(np.std(transforms, axis=0))


def orthogonal_mse_error(rots1, rots2) -> float:
    """Compute the MSE between two sets of orthogonal transformations.

    Computes MSE up to a global transformation.

    Args:
        rots1: First list of orthogonal matrices
        rots2: Second list of orthogonal matrices

    Returns:
        Mean squared error between rotation matrices
    """
    dim = len(rots1[0])
    rots1 = np.asarray(rots1)
    rots1 = rots1.transpose((0, 2, 1))
    rots2 = np.asarray(rots2)
    combined = np.mean(rots1 @ rots2, axis=0)
    _, s, _ = sp.linalg.svd(combined)
    return 2 * (dim - np.sum(s))


def _cov_svd(coordinates1: np.ndarray, coordinates2: np.ndarray):
    """Compute SVD of covariance matrix between two sets of coordinates.

    Args:
        coordinates1: First set of coordinates (array-like)
        coordinates2: Second set of coordinates (array-like)

    Returns:
        SVD decomposition (u, s, vh) of the covariance matrix

    Note:
        The two sets of coordinates need to have the same shape.
    """
    coordinates1 = coordinates1 - coordinates1.mean(axis=0)
    coordinates2 = coordinates2 - coordinates2.mean(axis=0)
    cov = coordinates1.T @ coordinates2
    return sp.linalg.svd(cov)


def relative_orthogonal_transform(
    coordinates1: np.ndarray, coordinates2: np.ndarray
) -> np.ndarray:
    """Find the best orthogonal transformation aligning two coordinate sets.

    This finds the optimal orthogonal transformation to align coordinates1 with 
    coordinates2 for the same nodes.

    Args:
        coordinates1: First set of coordinates (array-like)
        coordinates2: Second set of coordinates (array-like)

    Returns:
        Optimal orthogonal transformation matrix

    Note:
        The two sets of coordinates need to have the same shape.
        This is equivalent to the approach in "Closed-Form Solution of Absolute 
        Orientation using Orthonormal Matrices" (Journal of the Optical Society 
        of America A, July 1988).
    """
    u, _, vh = _cov_svd(coordinates1, coordinates2)
    return u @ vh


def nearest_orthogonal(mat) -> np.ndarray:
    """Compute nearest orthogonal matrix to a given input matrix.

    Args:
        mat: Input matrix

    Returns:
        Nearest orthogonal matrix
    """
    u, _, vh = sp.linalg.svd(mat)
    return u @ vh


def relative_scale(coordinates1: np.ndarray, coordinates2: np.ndarray, clamp=1e8):
    """Compute relative scale of two sets of coordinates for the same nodes.

    Args:
        coordinates1: First set of coordinates (array-like)
        coordinates2: Second set of coordinates (array-like)
        clamp: Maximum allowed scale, default is 1e8

    Returns:
        Relative scale factor

    Note:
        The two sets of coordinates need to have the same shape.
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