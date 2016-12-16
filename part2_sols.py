"""Part 2 solutions.

(!!!) DO NOT CHANGE ANY CODE IN THIS FILE. (!!!)
"""
import heapq
import os
import sys
import numpy as np
from corpus import Corpus

N_LARGEST = 4
VERBOSE = False

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
    best = []
    for i in range(k):
        heapq.heappush(best, (-float("inf"), (i, None)))

    for i in range(len(doc_matrix)):
        distance = -np.linalg.norm(doc_matrix[i] - doc_vector) ** 2
        end_priority, _ = best[0]
        if distance > end_priority and distance < -.01:
            heapq.heapreplace(best, (distance, (i, ids[i])))

    n_largest = heapq.nlargest(N_LARGEST, best)

    if VERBOSE:
        for i in range(N_LARGEST):
            print n_largest[i]
            print ids[n_largest[i][1][0]]
            print Corpus.document_text(n_largest[i][1][1])

    categories = []

    for i in range(k):
        categories.extend(Corpus.categories_of(best[i][1][1]))

    categories = set(categories)

    return categories

def test_knn(corp):
    """ Tests the knn classification function. Implement this however you like.

    Args:
        corpus: A corpus object, used for convenient access to the Reuters corpus.
    """
    ids, complete_matrix = corp.complete_matrix(include_ids=True)
    category_map = corp.complete_matrix_dict()
    categories = category_map.keys()
    errors = 0
    document_count = 0

    for _, category in enumerate(categories):
        documents_in_category = category_map[category]
        for i in range(len(documents_in_category)):
            print document_count
            document_count += 1
            if not category in knn(complete_matrix, ids, documents_in_category[i], 10):
                errors += 1
        print (errors, float(errors) / document_count, category)

if __name__ == '__main__':
    test_knn(Corpus())
