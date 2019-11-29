from services.index import positional_indexer
from collections import Counter, defaultdict
from math import log, sqrt
from services.document_manager import document_base
from functools import lru_cache

def get_tf(freq, type):
    if type == 'b':
        return 1 if freq > 0 else 0
    elif type == 'n':
        return freq
    elif type == 'l':
        return 1 + log(freq) if freq > 0 else 0
    else:
        raise NotImplementedError()

def get_idf(term, type):
    if not positional_indexer.inverted_index:
        return 0
    if type == 'n':
        return 1
    elif type == 't':
        return log(len(document_base)/len(positional_indexer.get_posting_list(term)))
    else:
        raise NotImplementedError()

def get_related_documents(query, k):
    terms = set(query)
    score = defaultdict(int)
    for t in terms:
        tf_q = get_tf(query.count(t), 'l')
        idf_q = get_idf(t, 't')
        w_q = tf_q*idf_q
        posting_list = positional_indexer.get_posting_list(t)
        for d, pos in posting_list:
            tf_d = get_tf(len(pos), 'l')
            idf_d = get_idf(t, 'n')
            w_d = tf_d*idf_d
            if not d in score:
                score[d] = 0
            score[d] += w_q*w_d
    for d in list(score):
        score[d] /= vector_length(d)
    counter = Counter(score)
    counter = counter.most_common(min(len(counter), k))
    bests = [rec[0] for rec in counter]
    return [' '.join(document_base[doc_id-1]) for doc_id in bests]

@lru_cache()
def vector_length(doc_id):
    document = document_base[doc_id-1]
    terms = set(document)
    res = 0
    for t in terms:
        tf_d = get_tf(document.count(t), 'l')
        idf_d = get_idf(t, 'n')
        w_d = tf_d*idf_d
        res += w_d*w_d
    return sqrt(res)
