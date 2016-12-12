"""
    Welcome to the search engine toolbox. Here, you will implement functions
    necessary for search functionality.

    Please implement each function as instructed to do so in the lab document.

    ALL YOUR WORK WILL BE COMPLETED IN THIS FILE. DO NOT CHANGE OTHER FILES.
"""
import numpy as np

def k_rank_approximate(doc_matrix, k):
    """Finds a k-rank approximation of the document vectors.

    Args:
        doc_matrix: An (n, m) matrix where each row is a document vector.
        k: The rank to approximate.

    Returns:
        An (n, m) matrix with rank k formed by the first k singular
        values/vectors of doc_matrix.
    """
    if np.linalg.matrix_rank(doc_matrix) <= k:
        raise Exception('Given document matrix is too low rank.')
    u, s, v = np.linalg.svd(doc_matrix)
    s = np.diag(s)
    return  np.dot(np.dot(u[:, :k], s[:k, :k]), v[:k, :])

def train_ls_classifier(A, y):
    """Trains the document category classifier by minimizing L1-regularized least squares loss.

    Args:
        A: An (n, m) matrix of training data where the rows are document vectors.
        y: A (n, 1) vector of labels (+1 or -1).

    Returns:
        An (m, 1) vector of the optimal classification weights.
    """
    A = np.mat(A)
    w = np.zeros((A.shape[1], 1))
    num_iterations = 20000
    learning_rate = 0.003
    regularization_weight = 0.1
    for i in range(num_iterations):
        z = np.asarray(w - learning_rate * A.transpose() * (A * w - y))
        w = np.sign(z) * np.maximum(abs(z) - (learning_rate * regularization_weight / 2), 0)
    return w

def train_svm_classifier(train_data, train_categories):
    """Trains the document category classifer by minimizing the hinge loss."""
    print 'Called train_svm_classifier!'
    return []

def classification_error(A, w, y):
    """Calculates the classification error for the given classifier weights.

    Args:
        A: An (n, m) matrix of test n m-dimensional examples
        w: An (n, 1) vector of learned weights for classification.
        y: Actual labels (+1 or -1) for each of the n examples.

    Returns:
        Number of misclassified examples.
    """
    print 'In classification_error!'
    return 0

def create_category_classifiers(category_train_documents):
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
            A = category_train_documents['cat_1'] + category_train_documents['cat_2']
            y = np.concatenate(
                np.ones(category_train_documents[cat_1]),
                -np.ones(category_train_documents[cat_2]))
            w = train_ls_classifier(A, y)
            category_classifiers[cat_1][cat_2] = w
            print w
            return {}
    return {}

def test_category_classifiers(category_classifiers, category_test_documents):
    """Tests the given classifier by calculating the classification error."""
    print 'Called evaluate_classifier!'
    return {}

def find_closest_documents(query, category_classifiers, category_docs):
    """Finds the document that best matches the query vector."""
    print 'Called find_closest_documents!'
    return []

def visualize(classifier):
    """Plot the classifier as specified in the assignment document."""
    print 'In visualize!'
