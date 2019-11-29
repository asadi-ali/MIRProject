import os
from copy import deepcopy
import nltk
from collections import defaultdict
from services import file_manager


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
        self.file_address = 'data/indices/positional.txt'

    def reload(self):
        self.inverted_index = defaultdict(list)
        self.deleted_docs = []

    def save(self):
        file_manager.save(self.file_address, self.inverted_index)
        file_manager.save_uncompress('data/indices/uncompress.txt',
                                     self.inverted_index)
        file_manager.save_gamma('data/indices/gamma.txt', self.inverted_index)
        file_manager.save_variable('data/indices/variable.txt',
                                   self.inverted_index)

    def load(self):
        if os.path.exists(self.file_address):
            self.inverted_index.update(
                file_manager.load(self.file_address))

            return True
        return False

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
        self.file_address = 'data/indices/bigram.txt'

    def reload(self):
        self.inverted_index = defaultdict(set)
        self.deleted_docs = []

    def save(self):
        index = deepcopy(dict(self.inverted_index))
        for k in index.keys():
            index[k] = list(index[k])

        file_manager.save(self.file_address, index)

    def load(self):
        if os.path.exists(self.file_address):
            index = file_manager.load(self.file_address)
            for k in index.keys():
                self.inverted_index[k] = set(index[k])
            return True
        return False

    def add_document(self, doc_id, words):
        for word in words:
            for cc in nltk.ngrams(word, n=2):
                self.inverted_index[''.join(cc)].add(word)

    def delete_document(self, doc_id):
        self.deleted_docs.append(doc_id)


positional_indexer = PositionalIndexer()
bigram_indexer = BigramIndexer()
