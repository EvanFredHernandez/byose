"""
    Main script for creating and interacting with the search engine.

    (!!!) DO NOT CHANGE ANY CODE IN THIS FILE. (!!!)
"""

from search_engine import SearchEngine, get_document

print 'Initializing BYOSE (this may take a minute)...'
se = SearchEngine()

print 'Welcome to BYOSE, a search engine for the Reuters Corpus!'

print '\n\nTo get started, enter a command:'
print '\t-- i\tinitalize the search engine'
print '\t-- s\tsearch'
print '\t-- q\tquit'

while True:
    command = raw_input('>> ')
    if command == 'i':
        se.approx_doc_matrices()
        se.train_classifiers()
    elif command == 's':
        results = se.search(raw_input('Query: '))
        for (i, result) in enumerate(results):
            print '\t[', i+1, '] ', result[i][1]
        while True:
            doc_num = int(raw_input('Number of document to open, or 0 to exit: '))
            if doc_num == 0:
                break
            elif doc_num > 0 and doc_num <= len(results):
                print get_document(results[doc_num][0])
                break
            else:
                print 'Bad article number. Try again.'
    elif command == 'q':
        break
    else:
        print 'Invalid command. Try again.'

