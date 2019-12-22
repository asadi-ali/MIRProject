from services.document_manager import import_documents
from services.index import positional_indexer, bigram_indexer
from functions import name_to_function_mapping
from services.classify import *


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


def learn():
    X_train, y_train = load_labeled_data('data/phase2_train.csv')
    clf = get_trained_classifier(X_train, y_train, 'random_forest')
    X_test, y_test = load_labeled_data('data/phase2_test.csv')
    pred = clf.predict_proba(X_test)
    print(y_test[0], pred[0])