"""Provides utility for interacting with the Reuters-21578 Corpus.

For those completing the lab: you will use this file to help test your solutions.
Some example usage is written out below, but feel free to peruse the code
and to use any public functions that seem useful.


(!!!) DO NOT CHANGE ANY CODE IN THIS FILE. (!!!)
"""
import pre
import numpy as np
from nltk.corpus import reuters

class Corpus(object):
    """Wrapper around the Reuters-21578 corpus with vectorization utility."""

    def __init__(self):
        """Loads the vectorized corpus."""
        corpus = pre.precompute_vectorized_corpus()
        self._vectorizer = corpus[0]
        self._train_docs = corpus[1]
        self._test_docs = corpus[2]

    @staticmethod
    def document_text(doc_id):
        """Returns the raw document text."""
        return reuters.raw(doc_id)

    @staticmethod
    def categories_of(doc_id):
        """Returns the document's category."""
        return reuters.categories(doc_id)

    @staticmethod
    def all_categories():
        """Returns all categories in the Reuters corpus."""
        return reuters.categories()

    def document_vector(self, doc_id):
        """Returns the vectorized document with doc_id. Excepts if ID doesn't exist."""
        if doc_id in self._train_docs.keys():
            return self._train_docs[doc_id]
        elif doc_id in self._test_docs.keys():
            return self._test_docs[doc_id]
        else:
            raise Exception('No document with ID: ' + doc_id)

    def vectorize(self, text):
        """Returns vectorized version of the text."""
        return self._vectorizer.transform([text])

    def complete_matrix(self, category=None, include_ids=False):
        """Returns the complete document matrix for the given category.

        A complete document matrix in which each row is a document vector
        belonging to that category, and which includes BOTH the documents
        belonging to the training set and the testing set of the corpus.

        Args:
            category (optional): The category to get the training matrix for.
                If not provided, returns the matrix of all documents in the corpus.
            include_ids (optional): If true, returns a 2-tuple where the first element is an
                array of document IDs and the second element is the category matrix. The first
                ID in the array of IDs corresponds to the first document in the matrix, etc.

        Returns:
            If category provided, returns matrix of all vectorized docs for that category.
            If category not provided, returns the matrix of all vectorized docs.
            If include_ids is true, tuples are returned instead of matrices, as described above.
        """
        return self._get_matrix(
            dict(self._train_docs.items() + self._test_docs.items()),
            category,
            include_ids)

    def complete_matrix_dict(self, include_ids=False):
        """Returns a map from category names to the corresponding complete category matrix.

        See the docstring of complete_matrix for info on what a complete document matrix is.

        Args:
            include_ids: If true, the values of the map are 2-tuples where the first
                element is an array of document IDs and the second element is the
                complete category matrix.

        Returns:
            Map given by
                    category --> complete_category_matrix
            or the map given by
                    category --> (doc_ids, complete_category_matrix)
            if include_ids is true.
        """
        return Corpus._get_matrix_dict(
            dict(self._train_docs.items() +  self._test_docs.items()), 
            include_ids)

    def train_matrix(self, category=None, include_ids=False):
        """Returns the matrix of vectorized training documents.

        A training matrix is one in which each row is a document vector taken
        from the set of training documents. Each document in a specific category's
        training matrix belongs to that category.

        See docstring for complete_matrix for parameter/return value details.
        """
        return Corpus._get_matrix(self._train_docs, category, include_ids)

    def train_matrix_dict(self, include_ids=False):
        """Returns a map from category names to the corresponding training matrix.

        See docstring for train_matrix for info on what training matrix is.

        See docstring for complete_matrix_dict for parameter/return value details.
        """
        return Corpus._get_matrix_dict(self._train_docs, include_ids)

    def test_matrix(self, category=None, include_ids=False):
        """Returns the matrix of vectorized testing documents.

        A testing matrix is one in which each row is a document vector taken
        from the set of testing documents. Each document in a specific category's
        testing matrix belongs to that category.

        See docstring for complete_matrix for parameter/return value details.
        """
        return Corpus._get_matrix(self._test_docs, category, include_ids)

    def test_matrix_dict(self, include_ids=False):
        """Returns a map from category names to the corresponding testing matrix.

        See docstring for test_matrix for info on what testing matrix is.

        See docstring for complete_matrix_dict for parameter/return value details.
        """
        return Corpus._get_matrix_dict(self._test_docs, include_ids)

    @staticmethod
    def _get_matrix(docs, category, include_ids):
        """Returns matrix of vectorized documents from given dictionary.

        See docstring for complete_matrix for parameter/return value details.
        """
        if category is None:
            ids = []
            matrix = []
            for doc_id, doc_vector in docs.items():
                ids.append(doc_id)
                matrix.append(doc_vector)
        elif not category in reuters.categories():
            raise Exception('No such category: ' + category)
        else:
            ids = []
            matrix = []
            for doc_id, doc_vector in docs.items():
                if category in reuters.categories(doc_id):
                    ids.append(doc_id)
                    matrix.append(doc_vector)
        matrix = np.mat(matrix)
        return (ids, matrix) if include_ids else matrix

    @staticmethod
    def _get_matrix_dict(docs, include_ids):
        """Returns a dict from categories to the corresponding given docs.
        
        See docstring for complete_matrix_dict for parameter/return value details.
        """
        docs_by_category = {category:([], []) for category in reuters.categories()}
        for doc_id, doc_vector in docs.items():
            for category in reuters.categories(doc_id):
                docs_by_category[category][0].append(doc_id)
                docs_by_category[category][1].append(doc_vector)
        docs_by_category = {cat:(tup[0], np.mat(tup[1]))
                            for cat, tup in docs_by_category.items()}
        return {cat:(tup if include_ids else tup[1])
                for cat, tup in docs_by_category.items()}
