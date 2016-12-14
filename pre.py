"""Functions for running expensive computations.

(!!!) DO NOT CHANGE ANY CODE IN THIS FILE. (!!!)
"""
from os.path import exists
import re
import dill

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import reuters, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

VECTORIZER_PATH = 'vectorizer.pkl'
TRAIN_PATH = 'train.pkl'
TEST_PATH = 'test.pkl'

def _tokenize(text):
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

def _tf_idf(docs):
    """Computes the tf-idf vectorizer for the given document set.

    Args:
        docs: List of documents on which to apply tf-idf.

    Returns:
        Instance of sklearn's TfidfVectorizer.
    """
    tfidf = TfidfVectorizer(tokenizer=_tokenize,
                            min_df=3, max_df=0.90,
                            max_features=3000, use_idf=True,
                            sublinear_tf=True, norm='l2')
    tfidf.fit(docs)
    return tfidf

def precompute_vectorized_corpus():
    """Precomputes the vectorizer and vectorizes the corpus.

    After running the first time, this initializer caches the vectorized
    Reuters corpus in a pickle file. To redo these computations, simply delete
    the vectorizer.pkl, train.pkl, and test.pkl files in the current directory.

    Returns:
        A tuple given by (vectorizer, train_docs, test_docs)
            vectorizer: TfidfVectorizer
            train_docs: Dict of the form {doc_id : doc_vector}
            test_docs: Dict of the form {doc_id : doc_vector}
    """
    if exists(VECTORIZER_PATH) and exists(TRAIN_PATH) and exists(TEST_PATH):
        with open(VECTORIZER_PATH) as vec, open(TRAIN_PATH) as train, open(TEST_PATH) as test:
            vectorizer = dill.load(vec)
            train_docs = dill.load(train)
            test_docs = dill.load(test)
        return (vectorizer, train_docs, test_docs)

    # Load the Reuters corpus.
    raw_train_docs = {}
    raw_test_docs = {}
    for doc_id in reuters.fileids():
        if doc_id.startswith('train'):
            raw_train_docs[doc_id] = reuters.raw(doc_id)
        else:
            raw_test_docs[doc_id] = reuters.raw(doc_id)

    # Before computing the vectorizer, check to see if it exists.
    if exists(VECTORIZER_PATH):
        with open(VECTORIZER_PATH, 'w') as vec:
            vectorizer = dill.load(vec)
    else:
        vectorizer = _tf_idf([doc for doc in raw_train_docs.values()])

    train_docs = {doc_id:vectorizer.transform([doc]).toarray()[0]
                  for doc_id, doc in raw_train_docs.items()}
    test_docs = {doc_id:vectorizer.transform([doc]).toarray()[0]
                 for doc_id, doc in raw_test_docs.items()}

    # Cache the vectorized corpus.
    with open(TRAIN_PATH, 'w') as train, open(TEST_PATH, 'w') as test:
        dill.dump(train_docs, train)
        dill.dump(test_docs, test)

    return (vectorizer, train_docs, test_docs)
