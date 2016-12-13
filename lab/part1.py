"""Part 1: Latent Semantic Analysis.

Implement each of the functions in this file as instructed in the lab document.
"""
import numpy as np

# TODO: Implement this method! 
def k_rank_approximate(doc_matrix, k):
    """Finds a k-rank approximation of the document vectors.

    Args:
        doc_matrix: An (n, m) matrix where each row is a document vector.
        k: The rank to approximate.

    Returns:
        An (n, m) matrix with rank k formed by the first k singular
        values/vectors of doc_matrix.
    """
    return  np.zeros(doc_matrix.shape)