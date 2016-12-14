sudo easy_install python pip
sudo pip install dill nltk sklearn numpy
python -m nltk.downloader punkt reuters stopwords
echo Preloading the vectorized Reuters-21578 corpus. This will take a while...
python -c 'import pre; pre.precompute_vectorized_corpus()'
