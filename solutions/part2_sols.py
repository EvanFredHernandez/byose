"""Part 2 solutions.

(!!!) DO NOT CHANGE ANY CODE IN THIS FILE. (!!!)
"""
import numpy as np

# TODO: Implement this method!
def k_nearest_neighbors(doc_matrix, k):
    """Clusters the documents into k distinct groups.

    Args:
        doc_matrix: An (n, m) matrix where each row is a vectorized document.
        k: The number of clusters to create.

    Returns:
        A list of clusters, where a cluster is a list of document vectors.
    """
    return [[]]