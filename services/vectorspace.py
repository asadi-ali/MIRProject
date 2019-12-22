from math import log
import numpy as np
from collections import defaultdict
from services.index import positional_indexer

def tf_prime(tf, type):
    if type == 'b':
        return 1 if tf > 0 else 0
    elif type == 'n':
        return tf
    elif type == 'l':
        return 1 + log(tf) if tf > 0 else 0
    else:
        raise NotImplementedError()

def idf_prime(df, N, type):
    if type == 'n':
        return 1
    elif type == 't':
        return log(N/df)
    else:
        raise NotImplementedError()

def get_idf(term, N, type):
    if not positional_indexer.inverted_index[term]:
        return 0
    return idf_prime(len(positional_indexer.get_posting_list(term)), N, type)

def get_tf(term, doc, type):
    return tf_prime(doc.count(term), type)

def doc_to_vec(document, N, mnemonic={'tf':'n', 'idf':'t', 'norm':'n'}):
    dictionary = positional_indexer.get_all_words()
    vec = []
    for term in dictionary:
        tf = get_tf(term, document, mnemonic['tf'])
        idf = get_idf(term, N, mnemonic['idf'])
        vec.append(tf * idf)
    return np.asarray(vec)
