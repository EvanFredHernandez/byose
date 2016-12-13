"""Part 1 solutions."""
import numpy as np

def k_rank_approximate(doc_matrix, k):
    """Finds a k-rank approximation of the document vectors.

    Args:
        doc_matrix: An (n, m) matrix where each row is a document vector.
        k: The rank to approximate.

    Returns:
        An (n, m) matrix with rank k formed by the first k singular
        values/vectors of doc_matrix.
    """
    if np.linalg.matrix_rank(doc_matrix) <= k:
        raise Exception('Given document matrix is too low rank.')
    u, s, v = np.linalg.svd(doc_matrix)
    s = np.diag(s)
    return  np.dot(np.dot(u[:, :k], s[:k, :k]), v[:k, :])