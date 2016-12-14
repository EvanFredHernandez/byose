"""Run me to test the search engine!

(!!!) DO NOT CHANGE ANY CODE IN THIS FILE. (!!!)
"""
from corpus import Corpus

from search_engine import SearchEngine

def main():
    """Runs simple command line interface for searching Reuters corpus."""
    print 'Initializing BYOSE (this may take a minute the first time)...'

    engine = SearchEngine()

    print """
       dBBBBBBBBBBBBBBBBBBBBBBBBb
      BP YBBBBBBBBBBBBBBBBBBBBBBBb
     dB   YBb  ~* REUTERS *~  YBBBb
     dB    YBBBBBBBBBBBBBBBBBBBBBBBb
      Yb    YBBBBBBBBBBBBBBBBBBBBBBBb
       Yb    YBBBBBBBBBBBBBBBBBBBBBBBb        Welcome to BYOSE, a search engine for the Reuters Corpus!
        Yb    YBBBBBBBBBBBBBBBBBBBBBBBb
         Yb    YBBBBBBBBBBBBBBBBBBBBBBBb      Created by Evan Hernandez, David Liang, and Alec Yu.
          Yb    YBBBBBBBBBBBBBBBBBBBBBBBb
           Yb   dBBBBBBBBBBBBBBBBBBBBBBBBb
            Yb dP=======================/
             YbB=======================(
              Ybb=======================\\
               Y888888888888888888DSI8888b"""

    print '\n\nTo get started, enter a command.'

    while True:
        print 'Commands:'
        print '\t-- i\tinitalize the search engine'
        print '\t-- s\tsearch'
        print '\t-- q\tquit'
        command = raw_input('>> ')
        if command == 'i':
            print 'Approximating document matrices with latent semantic analysis...'
            engine.approx_doc_matrices()

            print 'Training classifiers...'
            engine.train_classifiers()
        elif command == 's':
            results = engine.search(raw_input('Query: '))
            for (i, result) in enumerate(results):
                print '\t[', i+1, '] ', result[i][1]
            while True:
                doc_num = int(raw_input('Number of document to open, or 0 to exit: '))
                if doc_num == 0:
                    break
                elif doc_num > 0 and doc_num <= len(results):
                    print Corpus.document_text(results[doc_num][0])
                    break
                else:
                    print 'Bad article number. Try again.'

        elif command == 'q':
            break

        else:
            print 'Invalid command. Try again.'

if __name__ == '__main__':
    main()
