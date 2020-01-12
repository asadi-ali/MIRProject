from services.document_manager import import_documents
from services.index import positional_indexer, bigram_indexer
from functions import name_to_function_mapping
import services.classify as clf
import services.cluster as clt

def initialize():
    print("Initializing")
    #if not positional_indexer.load() or not bigram_indexer.load():
    positional_indexer.reload()
    bigram_indexer.reload()
    import_documents()
    positional_indexer.save()
    bigram_indexer.save()
    print("Initialization done.")


def serve():
    try:
        while True:
            method, *args = input().split(' ')
            if method in name_to_function_mapping:
                name_to_function_mapping[method](*args)
    except KeyboardInterrupt:
        print("The program is terminated.")


def learn(clf_type):
    clf.load_train_data('data/phase2_train.csv')
    clf.load_test_data('data/phase2_test.csv')
    clf.train_classifier(clf_type)
    clf.classify_documents()

def load_to_cluster(rep_type):
    clt.load_data('data/phase3.csv', rep_type, n_comps=50, scale=True)

def cluster(clt_type, path=None):
    clt.train_classifier(clt_type)
    clt.cluster_data(path + '.csv')
    clt.visualize(path + '.png')