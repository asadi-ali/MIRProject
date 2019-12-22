import os

from services.index import positional_indexer
from services.spell_correction import get_corrected_word
from services.search import get_related_documents, get_proximity_related_documents
from services.document_manager import process_farsi_document, process_english_document, get_farsi_commons, get_english_commons, doc_indices_by_type, document_type
from services.classify import classify_document, train_classifier, report_on_train_data, report_on_test_data, classify_documents

def show_posting_list(*args):
    posting_list = positional_indexer.get_posting_list(args[0])
    for doc_detail in posting_list:
        print("Doc: %s" % doc_detail[0])
        print("Positions: %s" % doc_detail[1])


def correct_query(*args):
    print([get_corrected_word(word) for word in args])

def get_variable_difference(*_args):
    print("Uncompressed: " + str(os.path.getsize('data/indices/uncompress.txt')))
    print("Variable Byte: " + str(os.path.getsize('data/indices/variable.txt')))


def get_gamma_difference(*_args):
    print("Uncompressed: " + str(os.path.getsize('data/indices/uncompress.txt')))
    print("Gamma: " + str(os.path.getsize('data/indices/gamma.txt')))

def search_for_document(*args):
    query = [get_corrected_word(word) for word in args]
    query_mnemonic = {'tf':'l', 'idf':'t', 'norm':'n'}
    doc_mnemonic = {'tf':'l', 'idf':'n', 'norm':'c'}
    related_documents = get_related_documents(query, 10, query_mnemonic, doc_mnemonic)
    print("Related Documents: ")
    for i, document in enumerate(related_documents):
        print(i, '.', document)

def search_for_document_by_type(*args):
    type = args[0]
    args = args[1:]
    query = [get_corrected_word(word) for word in args]
    query_mnemonic = {'tf': 'l', 'idf': 't', 'norm': 'n'}
    doc_mnemonic = {'tf': 'l', 'idf': 'n', 'norm': 'c'}
    related_documents = get_related_documents(query, 10, query_mnemonic, doc_mnemonic, doc_indices_by_type[type])
    print("Related Documents in %s: " % type)
    for i, document in enumerate(related_documents):
        print(i, '.', document)

def classify_query(*args):
    query = [get_corrected_word(word) for word in args]
    print(document_type[classify_document(query)])

def switch_classifier(*args):
    clf_type = args[0]
    train_classifier(clf_type)
    classify_documents()


def classifier_report(*args):
    print("Report on training data:")
    report_on_train_data()
    print("Report on test data:")
    report_on_test_data()

def proximity_search(*args):
    window = int(args[0])
    query = [get_corrected_word(word) for word in args[1:]]
    query_mnemonic = {'tf': 'l', 'idf': 't', 'norm': 'n'}
    doc_mnemonic = {'tf': 'l', 'idf': 'n', 'norm': 'c'}
    related_documents = get_proximity_related_documents(query, window, 10, query_mnemonic, doc_mnemonic)
    print("Proximity Related Documents: ")
    for i, document in enumerate(related_documents):
        print(i, '.', document)

def print_common_en(*args):
    print(get_english_commons())

def print_common_fa(*args):
    print(get_farsi_commons())

def process_text_fa(*args):
    print(process_farsi_document(' '.join(args)))

def process_text_en(*args):
    print(process_english_document(' '.join(args)))


name_to_function_mapping = {
    'show-posting-list': show_posting_list,
    'correct-query': correct_query,
    'get-variable-difference': get_variable_difference,
    'get-gamma-difference': get_gamma_difference,
    'search-for-document': search_for_document,
    'search-for-document-by-type': search_for_document_by_type,
    'classify-query': classify_query,
    'classifier-report': classifier_report,
    'switch-classifier': switch_classifier,
    'proximity-search': proximity_search,
    'print-common-en': print_common_en,
    'print-common-fa': print_common_fa,
    'process-text-fa': process_text_fa,
    'process-text-en': process_text_en,
}
