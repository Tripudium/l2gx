from .geo.geoalign import GeoAlignmentProblem
from .geo2.geoalign2 import GeoAlignmentProblem2
from .l2g.local2global import L2GAlignmentProblem
from .alignment import AlignmentProblem
from .errors import (
    procrustes_error,
    local_error,
    transform_error,
    orthogonal_MSE_error,
)
from .utils import relative_scale
from .registry import register_aligner, get_aligner

__all__ = [
    "GeoAlignmentProblem",
    "GeoAlignmentProblem2",
    "L2GAlignmentProblem",
    "AlignmentProblem",
    "procrustes_error",
    "local_error",
    "transform_error",
    "orthogonal_MSE_error",
    "relative_scale",
    "register_aligner",
    "get_aligner",
]

# pylint: disable=too-many-arguments
