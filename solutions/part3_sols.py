"""Part 3 solutions.

(!!!) DO NOT CHANGE ANY CODE IN THIS FILE. (!!!)
"""
import numpy as np

def train_ls_classifier(A, y):
    """Trains the document category classifier by minimizing L2-regularized least squares loss.

    Args:
        A: An (n, m) matrix of training data where the rows are document vectors.
        y: An (n, 1) vector of labels (+1 or -1).

    Returns:
        An (m, 1) vector of the optimal classification weights.
    """
    w = np.zeros((A.shape[1], 1))
    num_iterations = 10000
    learning_rate = 0.01
    regularization_weight = 0.1
    for i in range(num_iterations):
        z = w - learning_rate * A.transpose() * (A * w - y)
        w = (1/(1 + learning_rate * regularization_weight)) * z
    return w

def train_svm_classifier(A, y):
    """Trains the document category classifer by minimizing the hinge loss.

    Args:
        A: An (n, m) matrix of training data where the rows are document vectors.
        y:  An (n, 1) vector of labels (+1 or -1).

    Returns:
        An (m, 1) vector of the optimal classification weights.
    """
    w = np.zeros((A.shape[1], 1))
    num_iterations = 10000
    learning_rate = 0.01
    for i in range(num_iterations):
        # Compute the gradient.
        grad = np.zeros((A.shape[1], 1))
        for i in range(A.shape[0]):
            if y[i] * A[i] * w < 1:
                grad -= y[i, 0] * A[i, :].transpose()

        # Update the weight vector.
        w = w - learning_rate * grad
    return w

def classification_error(A, w, y):
    """Calculates the classification error for the given classifier weights.

    Args:
        A: An (n, m) matrix of test n m-dimensional examples
        w: An (n, 1) vector of learned weights for classification.
        y: Actual labels (+1 or -1) for each of the n examples.

    Returns:
        Number of misclassified examples.
    """
    return np.cumsum(0.5 * (np.sign(A * w) - y))

def train_one_vs_one_classifier(category_train_documents):
    """Computes a one-vs-one classifier schema for each pair of categories.

    Args:
        category_train_documents: A dict from categories to document matrices.

    Returns:
        A dict of dicts, where dict['category_1']['category_2'] is the weight vector
        that represents the one-vs-one classifier for category_1 and category_2.
    """
    category_classifiers = {}
    for cat_1 in category_train_documents.keys():
        category_classifiers[cat_1] = {}
        for cat_2 in [c for c in category_train_documents.keys() if c != cat_1]:
            A = np.mat(np.concatenate((
                category_train_documents[cat_1],
                category_train_documents[cat_2])))
            y = np.concatenate((
                np.ones((category_train_documents[cat_1].shape[0], 1)),
                -np.ones((category_train_documents[cat_2].shape[0], 1))))
            w = train_svm_classifier(A, y)
            category_classifiers[cat_1][cat_2] = w
    return category_classifiers

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