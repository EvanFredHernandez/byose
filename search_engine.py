"""
    Defines a simple search engine for querying the Reuters-21578 Corpus.
    This code is mostly logistical: it loads and tokenizes the dataset and
    depends on toolbox.py for any machine learning functionality.

    (!!!) DO NOT CHANGE ANY CODE IN THIS FILE. (!!!)
"""
from nltk import word_tokenize
from nltk.corpus import reuters, stopwords
from nltk.stem.porter import PorterStemmer
from numpy import sign
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import toolbox

class SearchEngine:
    def __init__(self):
        # Load the document data.
        self.train_docs = []
        self.test_docs = []
    
        for doc_id in reuters.fileids():
            if doc_id.startswith("train"):
                self.train_docs.append(reuters.raw(doc_id))
            else:
                self.test_docs.append(reuters.raw(doc_id))

        vectorizer = self.tf_idf(self.train_docs)
        train_docs_features = vectorizer.transform(self.train_docs)
        test_docs_features = vectorizer.transform(self.test_docs)
    
    def tf_idf(self, docs):
        tfidf = TfidfVectorizer(tokenizer=self.tokenize, min_df=3,
                            max_df=0.90, max_features=3000,
                            use_idf=True, sublinear_tf=True,
                            norm='l2')
        tfidf.fit(docs)
        return tfidf

    def tokenize(self, text):
        min_length = 3
        words = map(lambda word: word.lower(), word_tokenize(text))
        words = [word for word in words
                    if word not in stopwords.words("english")]
        tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
        p = re.compile('[a-zA-Z]+')
        filtered_tokens = list(filter(lambda token: p.match(token) and len(token)>=min_length, tokens))
        return filtered_tokens

    def search(self, query):
        print 'Query is ', query
        # TODO: Lol
