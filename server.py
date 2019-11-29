from services.import_documents import import_persian_documents, \
    import_english_documents
from functions import name_to_function_mapping


def initialize():
    import_english_documents('data/English.csv')
    import_persian_documents('data/Persian.xml')


def serve():
    try:
        while True:
            method, *args = input().split(' ')
            name_to_function_mapping[method](*args)
    except KeyboardInterrupt:
        print("The program is terminated.")
