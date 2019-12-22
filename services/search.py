from services.index import positional_indexer
from collections import Counter, defaultdict
from math import log, sqrt
from services.document_manager import document_base, raw_document_base
from functools import lru_cache
from .vectorspace import get_idf, get_tf, tf_prime

def get_related_documents(query, k, query_mnemonic, doc_mnemonic, candidates = None):
    terms = set(query)
    score = defaultdict(int)
    for t in terms:
        tf_q = get_tf(t, query, query_mnemonic['tf'])
        idf_q = get_idf(t, query_mnemonic['idf'])
        w_q = tf_q*idf_q
        posting_list = positional_indexer.get_posting_list(t)
        for d, pos in posting_list:
            if candidates is not None and not d in candidates:
                continue
            tf_d = tf_prime(len(pos), doc_mnemonic['tf'])
            idf_d = get_idf(t, doc_mnemonic['idf'])
            w_d = tf_d*idf_d
            score[d] += w_q*w_d
    if doc_mnemonic['norm'] == 'c':
        for d in list(score):
            score[d] /= vector_length(d)
    top_documents = Counter(score).most_common(min(len(score.keys()), k))
    return [' '.join(document_base[doc_id-1]) for doc_id, doc_score in top_documents]


def intersect(l1, l2):
    return [x for x in l1 if x in l2]

def get_proximity_related_documents(query, window, k, query_mnemonic, doc_mnemonic):
    terms = set(query)
    candidates = []
    first = True
    for t in terms:
        if first:
            candidates = [rec[0] for rec in positional_indexer.get_posting_list(t)]
            first = False
            continue
        candidates = intersect(candidates, [rec[0] for rec in positional_indexer.get_posting_list(t)])
    final_candidates = []
    for d in candidates:
        window_start = []
        for t in terms:
            appear = []
            for rec in positional_indexer.get_posting_list(t):
                if rec[0] == d:
                    appear = rec[1]
                    break
            window_start.extend(appear)
        final = True
        for p in window_start:
            pok = True
            for t in terms:
                ok = False
                for rec in positional_indexer.get_posting_list(t):
                    if rec[0] == d:
                        for ap in rec[1]:
                            if ap >= p and ap < p + window:
                                ok = True
                                break
                        break
                if not ok:
                    pok = False
                    break
            if pok:
                final = True
                break
        if final:
            final_candidates.append(d)
    return get_related_documents(query, k, query_mnemonic, doc_mnemonic, final_candidates)


@lru_cache()
def vector_length(doc_id):
    document = document_base[doc_id-1]
    terms = set(document)
    res = 0
    for t in terms:
        tf_d = tf_prime(document.count(t), 'l')
        idf_d = get_idf(t, 'n')
        w_d = tf_d*idf_d
        res += w_d*w_d
    return sqrt(res)
