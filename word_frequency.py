import re
from collections import Counter

def count_words(paragraph):
    # \b is a word boundary, \w+ matches 1 or more word characters
    # We convert to lower case before matching
    words = re.findall(r'\b\w+\b', paragraph.lower())
    
    # Counter returns a dictionary-like object mapping items to their counts
    return dict(Counter(words))

if __name__ == '__main__':
    text = "Hello world! This is a test. Hello, this test is only a test."
    print(count_words(text))
