# BYOSE
___
### Summary

A codelab for (B)uilding (Y)our (O)wn (S)earch (E)ngine: a self-contained introduction to text mining and document categorization. Readers will learn how to represent textual documents as vectors, reduce the dimensionality of document vectors in semantic space, and classify documents into predefined semantic categories using the k-nearest neighbors algorithm, regularized least squares classification, and regularized support vector machines.

This lab assumes some prior knowledge of linear algebra and pattern recognition. In particular, you should be familiar with: the singular value decomposition of a matrix, least squares classification, support vector machines, and regularization. 

### Setup

We used `Python 2.7.11` and `pip 9.0.1` on `OSX 10.0.1` but these versions aren't at all mandatory. If you do encounter trouble getting setup, we recommend that you try updating your packages.

### Manual setup

You will need to install each of the following.
* Python: see https://www.python.org/downloads/ 
* Pip: `sudo easy_install pip`
* Python libraries: `sudo pip install dill nltk sklearn numpy`
* NLTK corpora: `python -m nltk.downloader punkt stopwords reuters`
* This repository's code: `git clone git@github.com:EvanFredHernandez/byose.git`
