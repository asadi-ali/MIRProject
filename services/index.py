import nltk
from collections import defaultdict


class BaseIndexer(object):
    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def add_document(self, doc_id, words):
        raise NotImplementedError()

    def delete_document(self, doc_id):
        raise NotImplementedError()


class PositionalIndexer(BaseIndexer):
    def __init__(self):
        self.inverted_index = defaultdict(list)
        self.deleted_docs = []

    def save(self):
        pass

    def load(self):
        pass

    def add_document(self, doc_id, words):
        for i, word in enumerate(words):
            if (not self.inverted_index[word] or
                    self.inverted_index[word][-1][0] != doc_id):
                self.inverted_index[word].append((doc_id, []))
            self.inverted_index[word][-1][1].append(i)

    def delete_document(self, doc_id):
        self.deleted_docs.append(doc_id)

    def get_posting_list(self, word):
        return self.inverted_index[word]

    def get_all_words(self):
        return self.inverted_index.keys()


class BigramIndexer(BaseIndexer):
    def __init__(self):
        self.inverted_index = defaultdict(set)
        self.deleted_docs = []

    def save(self):
        pass

    def load(self):
        pass

    def add_document(self, doc_id, words):
        for word in words:
            if not word:
                continue
            for cc in nltk.ngrams(word, n=2):
                self.inverted_index[''.join(cc)].add(word)

    def delete_document(self, doc_id):
        self.deleted_docs.append(doc_id)


positional_indexer = PositionalIndexer()
bigram_indexer = BigramIndexer()
