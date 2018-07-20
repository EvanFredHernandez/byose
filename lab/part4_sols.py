"""Part 4 solutions. See part4.py for function docstrings.

(!!!) DO NOT CHANGE ANY CODE IN THIS FILE. (!!!)
"""
import numpy as np
import part3_sols as p3

def find_closest_documents(query, category_classifiers, category_docs, category_ids):
    best_docs = []
    query_categories = p3.classify(category_classifiers, query)
    for category in query_categories:
        similarities = category_docs[category] * query
        best_docs.append(category_ids[category][np.argmax(similarities)])
    return best_docs
