"""
    Welcome to the search engine toolbox. Here, you will implement functions
    necessary for search functionality. 

    Please implement each function as instructed to do so in the lab document.

    ALL YOUR WORK WILL BE COMPLETED IN THIS FILE.
"""

# Finds a k-rank approximation of the document vectors.
def k_rank_approximate(docs, k):
    print 'Called k_rank_approximate!'
    return []

# Trains the document category classifier by minimizing the least squares loss.
def train_ls_classifier(train_data, train_categories):
    print 'Called train_ls_classifier!'
    return []

# Trains the document category classifer by minimizing the hinge loss.
def train_svm_classifier(train_data, train_categories):
    print 'Called train_svm_classifier!'
    return []

def classification_error(w, test_data, test_categories):
    print 'In classification_error!'
    return 0

# Computes a classifier for each pair of categories.
def create_category_classifiers(category_train_documents):
    print 'Called one_vs_one_classifier!'
    return {}

# Tests the given classifier by calculating the classification error.
def test_category_classifiers(category_classifiers, category_test_documents):
    print 'Called evaluate_classifier!'
    return {}

# Finds the document that best matches the query vector.
def find_closest_documents(query, category_classifiers, category_docs):
    print 'Called find_closest_documents!'
    return []

# Plot the classifier as specified in the assignment document.
def visualize(classifier):
    print 'In visualize!'