from .geo.geoalign import GeoAlignmentProblem
from .l2g.local2global import L2GAlignmentProblem
from .alignment import (
    AlignmentProblem,
    procrustes_error,
    local_error,
    transform_error,
    orthogonal_MSE_error,
)
from .registry import register_aligner, get_aligner

__all__ = [
    "GeoAlignmentProblem",
    "L2GAlignmentProblem",
    "AlignmentProblem",
    "procrustes_error",
    "local_error",
    "transform_error",
    "orthogonal_MSE_error",
    "register_aligner",
    "get_aligner",
]

# pylint: disable=too-many-arguments
