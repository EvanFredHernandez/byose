"""Part 2 solutions.

(!!!) DO NOT CHANGE ANY CODE IN THIS FILE. (!!!)
"""
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
import corpus
import argparse

# TODO: Implement this method!
def unweighted_knn(doc_matrix, doc_vector, k):
    """Clusters the documents into k distinct groups.

    Args:
        doc_matrix: An (n, m) matrix where each row is a vectorized document.
        k: The number of clusters to create.
    """
    
    return 

# TODO: Implement this method!
def weighted_knn(doc_matrix, doc_vector, k):
    """Clusters the documents into k distinct groups.

    Args:
        doc_matrix: An (n, m) matrix where each row is a vectorized document.
        k: The number of clusters to create.
    """
    
    return

def test_unweighted_knn(corp):
    """ Tests the unweighted_knn classification function. Implement this however you like.

    Args: 
        corp: A Corpus object

    """
    data = corp.complete_matrix()
    print len(data)
    print data.shape

    return


def test_weighted_knn(corp):
    """ Tests the weighted_knn classification function. Implement this however you like.

    Args:
        corp: A Corpus object
    
    """
    data = corp.complete_matrix()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("weight", help="(w)eighted or (u)nweighted")
    args = parser.parse_args()

    corp = corpus.Corpus()

    if args.weight == "u":
        print "unweighted_knn"
        test_unweighted_knn(corp)
    elif args.weight == "w":
        test_weighted_knn(corp)
        print "weighted_knn"
    else:
        print "not an option"
