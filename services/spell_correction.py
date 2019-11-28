import nltk
from .index import positional_indexer, bigram_indexer


def get_corrected_word(word):
    if positional_indexer.get_posting_list(word):
        return word

    good_words = []
    for w in positional_indexer.get_all_words():
        if nltk.jaccard_distance(set(nltk.ngrams(w, n=2)),
                                 set(nltk.ngrams(word, n=2))) > 0.7:
            good_words.append(w)

    best_word = '###########################'
    for w in good_words:
        if nltk.edit_distance(w, word) < nltk.edit_distance(best_word, word):
            best_word = w

    return best_word
