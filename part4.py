"""Part 4: Putting it all together!

Implement each of the functions in this file as instructed in the lab document.
"""
import part3 as p3

# TODO: Implement this method!
def find_closest_documents(query, one_vs_one_classifier, docs_by_category):
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
    return []
