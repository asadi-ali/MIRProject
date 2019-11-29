from services.import_documents import import_persian_documents, import_english_documents
from services.index import positional_indexer, bigram_indexer


def initialize():
    import_english_documents('data/English.csv')
    import_persian_documents('data/Persian.xml')
    print(positional_indexer)


def serve():
    pass
