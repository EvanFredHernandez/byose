# BYOSE
___
<br>
### Summary

A codelab for (B)uilding (Y)our (O)wn (S)earch (E)ngine. This lab assumes some prior knowledge of linear algebra and pattern recognition. In particular, you should be familiar with: the singular value decomposition of a matrix, least squares classification and support vector machines, and regularization. 
<br>
<br>


### Setup

We used `Python 2.7.11` and `pip 9.0.1` on `OSX 10.0.1` but these versions aren't at all mandatory. If you do encounter trouble getting setup, we recommend that you try updating your packages.

After you've gotten those, run `setup.sh` to get setup. If this doesn't work, you can follow the below steps for manual setup.

### Manual setup
* Python:
	* Installation: https://www.python.org/downloads/ 
* Pip:
	* `sudo easy_install pip`
* Python libraries dill, nltk, sklearn, numpy:
	* `sudo pip install dill nltk sklearn numpy`
* NLTK corpora:
	* `python -m nltk.downloader punkt stopwords reuters`
* This repository's code:
	* `git clone git@github.com:EvanFredHernandez/byose.git`
	* Then run `python setup.py` to initialize BYOSE. It may take a few minutes.