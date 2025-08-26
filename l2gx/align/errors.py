"""
Error computation functions for alignment problems.

This module contains various error metrics used to evaluate the quality
of patch alignments and embedding recovery in the Local2Global algorithm.
"""

import numpy as np
import scipy as sp
from scipy.spatial import procrustes

from l2gx.patch import Patch


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
