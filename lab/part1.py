"""Part 1: Latent Semantic Analysis.

Implement each of the functions in this file as instructed in the lab document.
"""
from corpus import Corpus

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
    return []

# TODO: Implement this method!
def test_k_rank_approximate(corpus):
    """Test your k-rank approximation function.

    First, sanity check your function by finding a 2-rank approximation
    of the 4-dimensional identity. Then, find a 300 rank approximation
    of the acq category matrix and verify that it's rank is actually 300.

    Args:
        corpus: A corpus object, used for convenient access to the Reuters corpus.
    """
    return

if __name__ == '__main__':
    test_k_rank_approximate(Corpus())
