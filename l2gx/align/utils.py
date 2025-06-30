"""
Utils for alignment
"""

import numpy as np
import torch


def to_device(obj, device, dtype=torch.float32):
    """
    Recursively moves tensors in a nested structure to the specified device
    and converts them to the specified dtype.

    Args:
        obj: The object to process (tensor, dict, list, tuple, etc.)
        device: The target device ('cuda', 'cpu', etc.)
        dtype: The target data type (default: torch.float32)

    Returns:
        The same structure with all tensors moved to device and converted to dtype
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, dtype=dtype)
    if isinstance(obj, dict):
        return {k: to_device(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device, dtype) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(v, device, dtype) for v in obj)
    return obj


def get_intersections(patches, min_overlap=0):
    """Calculate the intersection of nodes between patches."""
    intersections = {}
    embeddings = {}
    for i, _ in enumerate(patches):
        for j in range(i + 1, len(patches)):
            intersections[(i, j)] = list(
                set(patches[i].nodes.tolist()).intersection(
                    set(patches[j].nodes.tolist())
                )
            )
            if len(intersections[(i, j)]) >= min_overlap:
                embeddings[(i, j)] = [
                    torch.tensor(
                        patches[i].get_coordinates(list(intersections[(i, j)]))
                    ),
                    torch.tensor(
                        patches[j].get_coordinates(list(intersections[(i, j)]))
                    ),
                ]
    # embeddings = list(itertools.chain.from_iterable(embeddings))
    return intersections, embeddings


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
