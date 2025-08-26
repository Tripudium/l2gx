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

from l2gx.graphs.tgraph import TGraph
from l2gx.align.registry import register_aligner
from l2gx.align.alignment import AlignmentProblem


@register_aligner("l2g")
class L2GAlignmentProblem(AlignmentProblem):
    """
    Implements the standard local2global algorithm using an unweighted patch graph
    """

    def __init__(
        self,
        randomized_method: str = "standard",  # "standard", "sparse_aware", "randomized"
        sketch_method: str = "gaussian",  # "gaussian", "rademacher", "fourier"
        verbose=False,
    ):
        """
        Initialise the alignment problem with a list of patches

        Args:
            verbose(bool): if True print diagnostic information (default: ``False``)
            randomized_method(str): method for eigenvalue decomposition ("standard", "sparse_aware", "randomized")
        """
        super().__init__(verbose=verbose)
        self.randomized_method = randomized_method
        self.sketch_method = sketch_method

    def align_patches(self, patch_graph: TGraph, use_scale: bool = True):
        """
        Align patches using Local2Global algorithm.

        Args:
            patch_graph: Pre-computed patch graph with patches as node features and overlap information
            scale: Whether to perform scale synchronization

        Returns:
            Self for method chaining
        """
        self._register_patches(patch_graph)

        # Special case for 2 patches - use direct Procrustes
        if self.n_patches == 2:
            if self.verbose:
                print("Using optimized Procrustes alignment for 2 patches")
            self._align_two_patches_procrustes(use_scale)
        else:
            # Standard L2G alignment for >2 patches
            if use_scale:
                self.scale_patches()
            self.rotate_patches(
                method=self.randomized_method, sketch_method=self.sketch_method
            )
            self.translate_patches()

        self._aligned_embedding = self.mean_embedding()
        return self

    def _align_two_patches_procrustes(self, use_scale: bool = True):
        """
        Optimized alignment for exactly 2 patches using direct Procrustes.

        This avoids eigenvalue decomposition issues and is more efficient
        for the simple 2-patch case.
        """
        from scipy.linalg import orthogonal_procrustes
        import numpy as np

        # Get the overlap nodes between the two patches
        overlap_key = (0, 1) if (0, 1) in self.patch_overlap else (1, 0)
        overlap_nodes = self.patch_overlap[overlap_key]

        if len(overlap_nodes) == 0:
            if self.verbose:
                print("Warning: No overlap between patches, skipping alignment")
            return

        # Get overlap indices for each patch
        patch0_nodes = np.array(self.patches[0].nodes)
        patch1_nodes = np.array(self.patches[1].nodes)

        # Find indices of overlap nodes in each patch
        overlap_idx0 = []
        overlap_idx1 = []

        for node in overlap_nodes:
            idx0 = np.where(patch0_nodes == node)[0]
            idx1 = np.where(patch1_nodes == node)[0]
            if len(idx0) > 0 and len(idx1) > 0:
                overlap_idx0.append(idx0[0])
                overlap_idx1.append(idx1[0])

        # Extract overlap embeddings
        X0 = self.patches[0].coordinates[overlap_idx0]
        X1 = self.patches[1].coordinates[overlap_idx1]

        if use_scale:
            # Compute scale factor
            scale0 = np.linalg.norm(X0, "fro") / X0.shape[0]
            scale1 = np.linalg.norm(X1, "fro") / X1.shape[0]
            scale_factor = scale0 / scale1
            X1_scaled = X1 * scale_factor
        else:
            scale_factor = 1.0
            X1_scaled = X1

        # Solve Procrustes problem: find R such that ||X1_scaled @ R - X0|| is minimized
        R, _ = orthogonal_procrustes(X1_scaled, X0)

        # Apply transformation to patch 1
        self.patches[1].coordinates = self.patches[1].coordinates * scale_factor @ R

        # Update transformation tracking
        self.scales[1] *= scale_factor
        self.rotations[1] = self.rotations[1] @ R

        # Translate patches to align their centroids on overlap
        self.translate_patches()

        if self.verbose:
            # Compute alignment error
            X1_aligned = self.patches[1].coordinates[overlap_idx1]
            error = np.linalg.norm(X0 - X1_aligned, "fro") / np.linalg.norm(X0, "fro")
            print(f"2-patch Procrustes alignment error: {error:.6f}")
