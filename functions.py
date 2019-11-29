from services.index import positional_indexer
from services.spell_correction import get_corrected_word


def show_posting_list(*args):
    posting_list = positional_indexer.get_posting_list(args[0])
    for doc_detail in posting_list:
        print("Doc: %s" % doc_detail[0])
        print("Positions: %s" % doc_detail[1])


def correct_query(*args):
    print([get_corrected_word(word) for word in args])


name_to_function_mapping = {
    'show-posting-list': show_posting_list,
    'correct-query': correct_query,
}
