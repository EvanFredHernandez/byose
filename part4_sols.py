"""Part 4 solutions.

(!!!) DO NOT CHANGE ANY CODE IN THIS FILE. (!!!)
"""
import numpy as np
import part3_sols as p3

def find_closest_documents(query, category_classifiers, category_docs, category_ids):
    """Finds the four documents that best match the query vector.

    For simplicity, this function categorizes the query and then finds the two
    best-matching documents in each category.

    Args:
        query: The vectorized query.
        one_vs_one_classifier: The one-vs-one classifier schema.
        category_docs: A dict that maps category names to document matrices.
        category_ids: An dict that maps category names to an array of ids,
            where the ith element in the array corresponds to the ith row of
            the category matrix.

    Returns:
        List of IDs of the four best-matching documents.
    """
    best_docs = []
    query_categories = p3.classify(category_classifiers, query)
    for category in query_categories:
        similarities = category_docs[category] * query
        best_docs.append(category_ids[category][np.argmax(similarities)])
    return best_docs
