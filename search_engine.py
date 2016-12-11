""" (!!!) DO NOT CHANGE ANY CODE IN THIS FILE. (!!!) """
import os
import re
import dill
import numpy as np
import toolbox as tb

from nltk import word_tokenize
from nltk.corpus import reuters, stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

VECTORIZER_PATH = 'vectorizer.pkl'
DOCS_PATH = 'docs.pkl'

def get_document(doc_id):
    """Returns the raw document text."""
    return reuters.raw(doc_id)

def get_category(doc_id):
    """Returns the document's category.'"""
    return reuters.categories(doc_id)

def as_matrix_dict(categories, docs):
    """Maps each category to a list of documents under that category.

    Args:
        docs: The documents to be grouped by category.
        categories: The corresponding categories for each doc.

    Returns:
        A map from category name to a matrix of documents in that category.
    """
    docs_by_category = {category:[] for category in reuters.categories()}
    for (i, doc) in enumerate(docs):
        for category in categories[i]:
            docs_by_category[category].append(doc)


            print type(doc)
            print len(doc)
            print doc
            return {}
    return {cat:np.mat(docs_by_category[cat]) for cat in docs_by_category.keys()}

class SearchEngine(object):
    """Defines a simple search engine for querying the Reuters-21578 Corpus.

    This code is mostly logistical: it loads and tokenizes the dataset and
    depends on toolbox.py for any machine learning functionality.

    Attributes:
        something...
        something else...
    """

    def __init__(self):
        """Loads the Reuters-21578 corpus from nltk and vectorizes the documents.

        After running the first time, this initializer caches the vectorized
        Reuters corpus in a pickle file. To redo these computations, simply delete
        the vectorizer.pickle and docs.pickle files in the current directory.
        """
        if os.path.exists(VECTORIZER_PATH) and os.path.exists(DOCS_PATH):
            with open(VECTORIZER_PATH) as vfile, open(DOCS_PATH) as dfile:
                self.vectorizer = dill.load(vfile)
                self.docs = dill.load(dfile)
            return

        # Load the Reuters corpus.
        train_docs = []
        test_docs = []
        train_categories = []
        test_categories = []
        for doc_id in reuters.fileids():
            if doc_id.startswith("train"):
                train_docs.append(get_document(doc_id))
                train_categories.append(get_category(doc_id))
            else:
                test_docs.append(get_document(doc_id))
                test_categories.append(get_category(doc_id))

        # Vectorize each document.
        self.vectorizer = self.tf_idf(train_docs)

        train_docs = self.vectorizer.transform(train_docs)
        test_docs = self.vectorizer.transform(test_docs)

        self.docs = {
            'train': as_matrix_dict(train_categories, train_docs),
            'test': as_matrix_dict(test_categories, test_docs)
        }
        self.category_classifiers = {}

        # Cache the vectorized corpus.
        with open(VECTORIZER_PATH, 'w') as vfile, open(DOCS_PATH, 'w') as dfile:
            dill.dump(self.vectorizer, vfile)
            dill.dump(self.docs, dfile)

    def tf_idf(self, docs):
        """Computes the tf-idf vectorizer for the given document set.

        Args:
            docs: List of documents on which to apply tf-idf.

        Returns:
            Instance of sklearn's TfidfVectorizer.
        """
        tfidf = TfidfVectorizer(tokenizer=self.tokenize,
                                min_df=3, max_df=0.90,
                                max_features=3000, use_idf=True,
                                sublinear_tf=True, norm='l2')
        tfidf.fit(docs)
        return tfidf

    def tokenize(self, text):
        """Tokenizes the given text into non-stopword, non-numeric word stems.

        Args:
            text: The text to tokenize.

        Returns:
            Tokenized text with stopwords removed.
        """
        min_length = 3
        tokens = [PorterStemmer().stem(word.lower())
                  for word in word_tokenize(text)
                  if word not in stopwords.words('english')]
        ptrn = re.compile('[a-zA-Z]+')
        return [token for token in tokens if ptrn.match(token) and len(token) >= min_length]

    def approx_doc_matrices(self):
        """Performs latent semantic analysis on each category document matrix."""

        """
        for category in self.docs['train'].keys():
            print category, ' : ', self.docs['train'][category].shape, ' vs ', len(reuters.fileids(category))
        return
        """
        self.docs['train'] = {
            category: tb.k_rank_approximate(self.docs['train'][category], 500)
            for category in reuters.categories()}
        # TODO: Should we also rank-approximate the test documents?
        # Should we store them together?

    def train_classifiers(self):
        """Delegates to the toolbox to train each category classifier."""
        self.category_classifiers = tb.create_category_classifiers(self.docs['train'])

    def test_classifiers(self):
        """Calculates classification error for each category classifier.

        Returns:
            A map from categories to classification error. For example:
            {
             'category_1': 5
             'category_2': 120
             'category_3': 26
             ...
            }

        Raises:
            Exception if classifiers have not yet been trained.
        """
        if not self.category_classifiers:
            raise Exception('You haven\'t trained the classifiers yet!')
        return tb.test_category_classifiers(self.category_classifiers, self.docs['test'])

    def search(self, query):
        """Finds 5 distinct documents that match the query.

        Args:
            query: The user's input query.

        Returns:
            A list of 5 (id, document) tuples that best match the query.

        Raises:
            Exception if classifiers not yet trained.
        """
        if not self.category_classifiers:
            raise Exception('Search engine not initialized.')
        return tb.find_closest_documents(
            self.vectorizer.transform(query),
            self.category_classifiers,
            {cat:[self.docs['train'][cat]] + [self.docs['test'][cat]]
             for cat in reuters.categories()})
        # TODO: This concatenation should be cached!

    def visualize(self):
        """Plots a visualization of the category classifiers."""
        tb.visualize(self.category_classifiers)
