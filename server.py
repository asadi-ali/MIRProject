from services.import_documents import import_documents
from functions import name_to_function_mapping


def initialize():
    print("Let's initialize.")
    import_documents()
    print("Initialization done.")


def serve():
    try:
        while True:
            method, *args = input().split(' ')
            name_to_function_mapping[method](*args)
    except KeyboardInterrupt:
        print("The program is terminated.")
