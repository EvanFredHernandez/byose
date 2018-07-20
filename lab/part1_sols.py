"""Part 1 solutions. See part1.py for function docstrings.

(!!!) DO NOT CHANGE ANY CODE IN THIS FILE. (!!!)
"""
import numpy as np
from corpus import Corpus

def k_rank_approximate(doc_matrix, k):
    if np.linalg.matrix_rank(doc_matrix) <= k:
        raise Exception('Given document matrix is too low rank.')
    u, s, v = np.linalg.svd(doc_matrix)
    s = np.diag(s)
    return  np.dot(np.dot(u[:, :k], s[:k, :k]), v[:k, :])

def test_k_rank_approximate(corpus):
    I4 = np.identity(4)
    I4_approx = k_rank_approximate(I4, 2)
    print 'First approximation rank:', np.linalg.matrix_rank(I4_approx)

    doc_matrix = corpus.complete_matrix('acq')
    doc_matrix_approx = k_rank_approximate(doc_matrix, 300)
    print 'Second approximation rank:', np.linalg.matrix_rank(doc_matrix_approx)

if __name__ == '__main__':
    test_k_rank_approximate(Corpus())
