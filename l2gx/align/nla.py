"""
Randomized Linear Algebra for Alignment Problems

This module provides randomized alternatives to standard eigendecomposition
methods used in the synchronization process. The methods are based on
randomized subspace iteration and related techniques.

References:
- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with
  randomness: Probabilistic algorithms for constructing approximate matrix
  decompositions. SIAM review, 53(2), 217-288.
- Musco, C., & Musco, C. (2015). Randomized block Krylov methods for stronger
  and faster approximate singular value decomposition. NIPS.
- https://arxiv.org/pdf/2002.01387 (Randomized Numerical Linear Algebra)
"""

import numpy as np
import scipy.sparse as ss
from typing import Optional


def rademacher_matrix(
    n: int, k: int, sparsity: float = 0.1, random_state: Optional[int] = None
) -> ss.spmatrix:
    """
    Generate sparse Rademacher (±1) random matrix.

    Rademacher matrices often perform as well as Gaussian while being
    faster to generate and apply (no floating point multiplication).

    Args:
        n: Number of rows
        k: Number of columns
        sparsity: Fraction of non-zero entries
        random_state: Random seed

    Returns:
        Sparse Rademacher matrix
    """
    if random_state is not None:
        np.random.seed(random_state)

    nnz = max(1, int(sparsity * n * k))

    rows = np.random.randint(0, n, size=nnz)
    cols = np.random.randint(0, k, size=nnz)

    # Rademacher random variables (±1)
    data = 2 * np.random.randint(0, 2, size=nnz) - 1
    data = data.astype(np.float64)

    matrix = ss.coo_matrix((data, (rows, cols)), shape=(n, k)).tocsr()

    # Normalize (optional - Rademacher often works well unnormalized)
    col_norms = np.sqrt(np.array(matrix.multiply(matrix).sum(axis=0)).flatten())
    col_norms = np.maximum(col_norms, 1e-12)
    col_scaling = ss.diags(1.0 / col_norms, shape=(k, k))
    matrix = matrix @ col_scaling

    return matrix


def randomized_synchronise(
    matrix: ss.spmatrix,
    blocksize: int = 1,
    symmetric: bool = True,
    n_iter: int = 4,
    n_oversamples: int = 10,
    power_iterations: int = 0,
    random_state: Optional[int] = None,
    sketch_method: str = "gaussian",
    verbose: bool = False,
) -> np.ndarray:
    """
    Synchronisation using randomized subspace iteration.

    This function computes the leading k eigenvectors of a symmetric matrix
    using randomized methods.

    Args:
        matrix: Sparse matrix (assumed symmetric when symmetric=True)
        blocksize: Number of eigenvectors to compute (k in the original)
        symmetric: Must be True for this implementation
        n_iter: Number of subspace iterations (default: 4)
        n_oversamples: Additional dimensions for oversampling (default: 10)
        power_iterations: Number of power iterations for stability (default: 0)
        random_state: Random seed for reproducibility
        verbose: Print diagnostic information

    Returns:
        Eigenvectors reshaped as (dim // blocksize, blocksize, blocksize)

    Raises:
        ValueError: If symmetric=False (not implemented)
        ValueError: If blocksize > matrix dimension
    """
    if not symmetric:
        raise ValueError("Randomized synchronise only implemented for symmetric=True")

    if random_state is not None:
        np.random.seed(random_state)

    dim = matrix.shape[0]
    if blocksize > dim:
        raise ValueError(
            f"blocksize ({blocksize}) cannot exceed matrix dimension ({dim})"
        )

    if verbose:
        print(
            f"Randomized synchronise: dim={dim}, blocksize={blocksize}, "
            f"n_iter={n_iter}, n_oversamples={n_oversamples}"
        )

    # Add identity shift for positive semi-definiteness (matching original)
    shifted_matrix = matrix + ss.eye(dim)

    # Use randomized subspace iteration
    eigs, vecs = randomized_eig(
        shifted_matrix,
        k=blocksize,
        n_iter=n_iter,
        n_oversamples=n_oversamples,
        power_iterations=power_iterations,
        sketch_method=sketch_method,
        verbose=verbose,
    )

    # Sort eigenvalues in descending order (matching original)
    order = np.argsort(eigs)[::-1]
    vecs = vecs[:, order]
    eigs = eigs[order]

    if verbose:
        print(f"eigenvalues: {eigs}")

    # Reshape to match original format
    vecs.shape = (dim // blocksize, blocksize, blocksize)
    return vecs


def randomized_eig(
    A: ss.spmatrix,
    k: int,
    n_iter: int = 4,
    n_oversamples: int = 10,
    power_iterations: int = 0,
    sketch_method: str = "gaussian",  # rademacher, gaussian, fourier
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute k largest eigenvalues and eigenvectors using randomized methods.

    This implements the randomized subspace iteration algorithm for symmetric
    matrices, as described in Halko et al. (2011) and related work.

    Algorithm outline:
    1. Generate random test matrix Omega of size n x (k + p)
    2. Form Y = A * Omega (range finding)
    3. Optionally apply power iterations: Y = (A^q) * Omega
    4. Compute QR decomposition: Y = Q * R
    5. Form small matrix B = Q^T * A * Q
    6. Compute eigendecomposition of B: B * V = V * Lambda
    7. Return eigenvectors: Q * V and eigenvalues Lambda

    Args:
        A: Symmetric sparse matrix
        k: Number of eigenvalues/eigenvectors to compute
        n_iter: Number of subspace iterations for refinement
        n_oversamples: Oversampling parameter (p in literature)
        power_iterations: Number of power iterations (q in literature)
        verbose: Print diagnostic information

    Returns:
        eigenvalues: Array of k largest eigenvalues
        eigenvectors: Matrix with k eigenvectors as columns
    """
    n = A.shape[0]
    target_rank = min(k + n_oversamples, n)

    if verbose:
        print(f"Randomized eigsh: n={n}, k={k}, target_rank={target_rank}")

    # Stage 1: Range finding with randomized sampling
    if sketch_method == "gaussian":
        Omega = np.random.randn(n, target_rank)
        Y = A @ Omega
    elif sketch_method == "rademacher":
        Omega_sparse = rademacher_matrix(n, target_rank)
        # Convert sparse result to dense for QR decomposition
        Y = A @ Omega_sparse.toarray()
    else:
        raise ValueError(f"Unknown method: {sketch_method}")

    # Ensure Y is 2D array for QR decomposition
    if Y.ndim != 2:
        raise ValueError(
            f"Matrix multiplication resulted in {Y.ndim}D array, expected 2D"
        )

    # Optional power iterations for better accuracy
    for _ in range(power_iterations):
        Y = A @ Y

    # QR decomposition for orthogonalization
    Q, _ = np.linalg.qr(Y)

    # Stage 2: Subspace iteration refinement
    for i in range(n_iter):
        if verbose and n_iter > 1:
            print(f"  Subspace iteration {i + 1}/{n_iter}")

        # Apply matrix and orthogonalize
        Y = A @ Q
        Q, _ = np.linalg.qr(Y)

    # Stage 3: Solve eigenvalue problem in reduced space
    # Form B = Q^T * A * Q (small k x k matrix)
    B = Q.T @ (A @ Q)

    # Ensure B is symmetric (numerical stability)
    B = 0.5 * (B + B.T)

    # Compute eigendecomposition of small matrix
    eigs_small, vecs_small = np.linalg.eigh(B)

    # Extract top k eigenvalues and eigenvectors
    idx = np.argsort(eigs_small)[::-1][:k]
    eigenvalues = eigs_small[idx]
    eigenvectors = Q @ vecs_small[:, idx]

    return eigenvalues, eigenvectors


def adaptive_randomized_synchronise(
    matrix: ss.spmatrix,
    blocksize: int = 1,
    symmetric: bool = True,
    target_accuracy: float = 1e-6,
    max_iter: int = 10,
    initial_oversamples: int = 5,
    sketch_method: str = "gaussian",
    verbose: bool = False,
) -> np.ndarray:
    """
    Adaptive version that automatically adjusts parameters for target accuracy.

    This version monitors convergence and adapts the oversampling and iteration
    parameters to achieve a target accuracy level.

    Args:
        matrix: Sparse matrix (assumed symmetric)
        blocksize: Number of eigenvectors to compute
        symmetric: Must be True
        target_accuracy: Target relative error for eigenvalues
        max_iter: Maximum number of adaptive iterations
        initial_oversamples: Starting oversampling parameter
        verbose: Print diagnostic information

    Returns:
        Eigenvectors in the same format as randomized_synchronise
    """
    if not symmetric:
        raise ValueError(
            "Adaptive randomized synchronise only implemented for symmetric=True"
        )

    dim = matrix.shape[0]
    shifted_matrix = matrix + ss.eye(dim)

    current_oversamples = initial_oversamples
    prev_eigs = None

    for iteration in range(max_iter):
        if verbose:
            print(
                f"Adaptive iteration {iteration + 1}: oversamples={current_oversamples}"
            )

        # Compute eigenvalues with current parameters
        eigs, vecs = randomized_eig(
            shifted_matrix,
            k=blocksize,
            n_iter=2,  # Start with fewer iterations
            n_oversamples=current_oversamples,
            sketch_method=sketch_method,
            verbose=False,
        )

        # Check convergence if we have previous results
        if prev_eigs is not None:
            rel_error = np.max(np.abs(eigs - prev_eigs) / (np.abs(prev_eigs) + 1e-16))
            if verbose:
                print(f"  Relative error: {rel_error:.2e}")

            if rel_error < target_accuracy:
                if verbose:
                    print(f"  Converged in {iteration + 1} iterations")
                break

        prev_eigs = eigs.copy()
        current_oversamples = min(current_oversamples + 5, dim - blocksize)

    # Sort in descending order
    order = np.argsort(eigs)[::-1]
    vecs = vecs[:, order]
    eigs = eigs[order]

    if verbose:
        print(f"Final eigenvalues: {eigs}")

    # Reshape to match original format
    vecs.shape = (dim // blocksize, blocksize, blocksize)
    return vecs


def randomized_synchronise_sparse_aware(
    matrix: ss.spmatrix,
    blocksize: int = 1,
    symmetric: bool = True,
    density_threshold: float = 0.1,
    sketch_method: str = "gaussian",
    verbose: bool = False,
) -> np.ndarray:
    """
    Sparse-aware version that adapts strategy based on matrix density.

    For very sparse matrices, this uses a different approach that's more
    efficient for sparse matrix-vector products.

    Args:
        matrix: Sparse matrix
        blocksize: Number of eigenvectors to compute
        symmetric: Must be True
        density_threshold: Threshold for considering matrix "dense"
        verbose: Print diagnostic information

    Returns:
        Eigenvectors in the same format as randomized_synchronise
    """
    if not symmetric:
        raise ValueError(
            "Sparse-aware randomized synchronise only implemented for symmetric=True"
        )

    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])

    if verbose:
        print(f"Matrix density: {density:.4f}")

    if density > density_threshold:
        # For denser matrices, use more oversampling
        n_oversamples = max(10, blocksize)
        n_iter = 3
    else:
        # For sparser matrices, use fewer oversamples but more iterations
        n_oversamples = max(5, blocksize // 2)
        n_iter = 6

    if verbose:
        print(f"Using n_oversamples={n_oversamples}, n_iter={n_iter}")

    return randomized_synchronise(
        matrix=matrix,
        blocksize=blocksize,
        symmetric=symmetric,
        n_iter=n_iter,
        n_oversamples=n_oversamples,
        sketch_method=sketch_method,
        verbose=verbose,
    )


def synchronise_eigs(
    matrix: ss.spmatrix,
    blocksize=1,
    symmetric=False,
    sketch_method="gaussian",
    verbose=False,
):
    """
    Standard synchronization method from AlignmentProblem.

    This is the original _synchronise method from the AlignmentProblem class,
    providing exact compatibility with the standard eigendecomposition approach.

    Args:
        matrix: Sparse matrix input
        blocksize: Number of eigenvectors to compute (default: 1)
        symmetric: Whether to treat matrix as symmetric (default: False)
        sketch_method: Ignored for standard method (included for API compatibility)
        verbose: Print diagnostic information (default: False)

    Returns:
        Eigenvectors reshaped as (dim // blocksize, blocksize, blocksize)
    """
    # Random number generator for synchronization (matching original)
    rg = np.random.default_rng()

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
    if verbose:
        print(f"eigenvalues: {eigs}")
    vecs.shape = (dim // blocksize, blocksize, blocksize)
    return vecs


# Convenience function that matches the original _synchronise interface exactly
def synchronise(
    matrix: ss.spmatrix,
    blocksize: int = 1,
    symmetric: bool = True,
    method: str = "standard",
    verbose: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Drop-in replacement for AlignmentProblem._synchronise using randomized methods.

    Args:
        matrix: Sparse matrix input
        blocksize: Number of eigenvectors to compute
        symmetric: Whether matrix is symmetric (randomized methods require True)
        method: Which randomized method to use ('standard', 'adaptive', 'sparse_aware')
        verbose: Print diagnostic information
        **kwargs: Additional arguments passed to specific methods

    Returns:
        Eigenvectors in the same format as original _synchronise

    Note:
        This function assumes symmetric=True. For non-symmetric matrices,
        consider using the original _synchronise method or implement
        randomized methods for general matrices.
    """
    method_map = {
        "randomized": randomized_synchronise,
        "adaptive": adaptive_randomized_synchronise,
        "sparse_aware": randomized_synchronise_sparse_aware,
        "standard": synchronise_eigs,
    }

    if method not in method_map:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {list(method_map.keys())}"
        )

    return method_map[method](
        matrix=matrix,
        blocksize=blocksize,
        symmetric=symmetric,
        verbose=verbose,
        **kwargs,
    )
