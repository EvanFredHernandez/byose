"""Part 1 solutions.

(!!!) DO NOT CHANGE ANY CODE IN THIS FILE. (!!!)
"""
import numpy as np
from corpus import Corpus

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

def test_k_rank_approximate(corpus):
    """Test your k-rank approximation function.

    First, sanity check your function by finding a 2-rank approximation
    of the 4-dimensional identity. Then, find a 300 rank approximation
    of the acq category matrix and verify that it's rank is actually 300.

    Args:
        corpus: A corpus object, used for convenient access to the Reuters corpus.
    """
    I4 = np.identity(4)
    I4_approx = k_rank_approximate(I4, 2)
    print 'First approximation rank:', np.linalg.matrix_rank(I4_approx)

    doc_matrix = corpus.complete_matrix('acq')
    doc_matrix_approx = k_rank_approximate(doc_matrix, 300)
    print 'Second approximation rank:', np.linalg.matrix_rank(doc_matrix_approx)

if __name__ == '__main__':
    test_k_rank_approximate(Corpus())
