from services.document_manager import import_documents
from services.index import positional_indexer, bigram_indexer
from functions import name_to_function_mapping


def initialize():
    #if not positional_indexer.load() or not bigram_indexer.load():
    positional_indexer.reload()
    bigram_indexer.reload()
    import_documents()
    positional_indexer.save()
    bigram_indexer.save()
    print("I'm ready!")


def serve():
    try:
        while True:
            method, *args = input().split(' ')
            name_to_function_mapping[method](*args)
    except KeyboardInterrupt:
        print("The program is terminated.")
