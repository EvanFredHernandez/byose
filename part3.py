"""Part 3: Supervised categorization of text documents.

Implement each of the functions in this file as instructed in the lab document.
"""
import numpy as np

# TODO: Implement this method!
def train_ls_classifier(A, y):
    """Trains the document category classifier by minimizing L2-regularized least squares loss.

    Args:
        A: An (n, m) matrix of training data where the rows are document vectors.
        y: An (n, 1) vector of labels (+1 or -1).

    Returns:
        An (m, 1) vector of the optimal classification weights.
    """
    return np.zeros(A.size[1], 1)

# TODO: Implement this method!
def train_svm_classifier(A, y):
    """Trains the document category classifer by minimizing the hinge loss.

    Args:
        A: An (n, m) matrix of training data where the rows are document vectors.
        y:  An (n, 1) vector of labels (+1 or -1).

    Returns:
        An (m, 1) vector of the optimal classification weights.
    """
    return np.zeros(A.size[1], 1)

# TODO: Implement this method!
def classification_error(A, w, y):
    """Calculates the classification error for the given classifier weights.

    Args:
        A: An (n, m) matrix of test n m-dimensional examples
        w: An (n, 1) vector of learned weights for classification.
        y: Actual labels (+1 or -1) for each of the n examples.

    Returns:
        Number of misclassified examples.
    """
    return 0

# TODO: Implement this method!
def train_one_vs_one_classifier(category_train_documents):
    """Computes a one-vs-one classifier schema for each pair of categories.

    Args:
        category_train_documents: A dict from categories to document matrices.

    Returns:
        A dict of dicts, where dict['category_1']['category_2'] is the weight vector
        that represents the one-vs-one classifier for category_1 and category_2.
    """
    return {}

# TODO: Implement this method!
def classify(one_vs_one_classifier, doc):
    """Given a one-vs-one classifier schema, classifies the example.

    Args:
        one_vs_one_classifier: The one-vs-one classifier schema.
        doc: The vectorized document to classify.

    Returns:
        A 3-tuple of the three majority-vote categories.
    """
    return ('best_category', 'second_best_category', 'third_best_category')