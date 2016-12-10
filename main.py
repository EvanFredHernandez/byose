"""
    Main script for creating and interacting with the search engine.

    (!!!) DO NOT CHANGE ANY CODE IN THIS FILE. (!!!)
"""

from search_engine import SearchEngine

print 'Initializing BYOSE (this will take a minute)...'
se = SearchEngine()

print 'Welcome to BYOSE, a search engine for the Reuters Corpus!'

print '\n\nTo get started, enter a command:'
print '\t-- q\tquit'
print '\t-- s\tsearch'

while True:
    command = raw_input('>> ')
    if command == 'q':
        break
    elif command == 's':
        query = raw_input('Query: ')
        se.search(query)
    else:
        print 'Invalid command. Try again.'    