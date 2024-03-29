import string

import hazm
import nltk
from .index import positional_indexer


def get_corrected_word(word):
    if word[0] in string.ascii_letters:
        stemmer = nltk.PorterStemmer()
    else:
        stemmer = hazm.Stemmer()
    word = stemmer.stem(word)

    if word in positional_indexer.inverted_index:
        return word

    good_words = []
    for w in positional_indexer.get_all_words():
        if nltk.jaccard_distance(set(nltk.ngrams(w, n=2)),
                                 set(nltk.ngrams(word, n=2))) < 0.3:
            good_words.append(w)

    best_word = '###########################'
    for w in good_words:
        if nltk.edit_distance(w, word) < nltk.edit_distance(best_word, word):
            best_word = w

    return best_word
