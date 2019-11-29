import os

from services.index import positional_indexer
from services.spell_correction import get_corrected_word


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


name_to_function_mapping = {
    'show-posting-list': show_posting_list,
    'correct-query': correct_query,
    'get-variable-difference': get_variable_difference,
    'get-gamma-difference': get_gamma_difference
}
