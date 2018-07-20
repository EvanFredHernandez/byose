"""Part 2: Unsupervised categorization of text documents.

Implement each of the functions in this file as instructed in the lab document.
"""
from corpus import Corpus

# TODO: Implement this method!
def knn(doc_matrix, ids, doc_vector, k):
    """Finds the k most similar docs in the doc matrix to the given doc vector.

    Args:
        doc_matrix: An (n, m) matrix where each row is a vectorized document.
        ids: An (n, 1) array where the ith element corresponds to the ith row in doc_matrix.
        doc_vector: An (m, 1) vector that represnts a vectorized document.
        k: The number of clusters to create.

    Returns:
        List of unique categories of the k nearest neighbors of the doc_vector
    """
    return []

# TODO: Implement this method!
def test_knn(corp):
    """ Tests the knn classification function. Implement this however you like.

    Args:
        corpus: A corpus object, used for convenient access to the Reuters corpus.
    """
    return

if __name__ == '__main__':
    test_knn(Corpus())
