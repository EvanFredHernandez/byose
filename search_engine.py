"""Defines a simple search engine for the Reuters-21578 corpus.

(!!!) DO NOT CHANGE ANY CODE IN THIS FILE. (!!!)
"""
import imp
import thread
from corpus import Corpus

p1 = imp.load_module('part1_sols', 'solutions/part1_sols')
p2 = imp.load_module('part2_sols', 'solutions/part2_sols')
p3 = imp.load_module('part3_sols', 'solutions/part3_sols')
p4 = imp.load_module('part4_sols', 'solutions/part4_sols')

class SearchEngine(object):
    """Simple search engine class for the Reuters-21578 corpus.

    This is a thin wrapper about the Database class that calls on functions
    from the lab exercises to implement search functionality.
    """

    def __init__(self):
        """Caches the vectorized corpus."""
        self.corp = Corpus()
        self.approx_docs = {'train': {}, 'test': {}}
        self.one_vs_one_classifier = None
        self.max_rank_reduction = 300

    def approx_doc_matrices(self):
        """Performs latent semantic analysis on each training category doc matrix.

        Each approximation is calculated on a separate thread.
        """
        for category in Corpus.all_categories():
            thread.start_new_thread(self._approx_doc_matrix, (category,))

    def _approx_doc_matrix(self, category):
        """Launches k-rank approximation for given category matrix."""
        train_matrix = self.corp.train_matrix(category)
        self.approx_docs['train'][category] = p1.k_rank_approximate(
            train_matrix, self._compute_rank_reduction(train_matrix))

    def _compute_rank_reduction(self, matrix):
        """Utility method for determining the appropriate rank reduction.

        Args:
            matrix: The (n, m) matrix to rank-reduce.abs

        Returns:
            The smaller of 1/3 * m and 300.
        """
        return min(matrix.shape[0] / 3, self.max_rank_reduction)

    def train_classifiers(self):
        """Delegates to the toolbox to train each category classifier."""
        self.one_vs_one_classifier = p3.train_one_vs_one_classifier(self.approx_docs['train'])

    def search(self, query):
        """Finds 5 distinct documents that match the query.

        Args:
            query: The user's input query.

        Returns:
            A list of 5 (id, document) tuples that best match the query.

        Raises:
            Exception if classifiers not yet trained.
        """
        return p4.find_closest_documents(
            self.corp.vectorize(query),
            self.one_vs_one_classifier,
            {cat:[self.approx_docs['train'][cat]] + [self.approx_docs['test'][cat]]
             for cat in Corpus.all_categories()})
