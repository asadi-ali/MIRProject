import csv
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from services.index import positional_indexer, bigram_indexer

doc_id = 0

def import_documents(file_address, language):
    global doc_id
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    stemer = PorterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()
    with open('data/English.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        title = True
        for row in reader:
            if title:
                title = False
                continue
            text = ' '.join(row, )
            document = tokenizer.tokenize(text)
            document = [word for word in document if not word in stop_words]
            document = [stemer.stem(word) for word in document]
            document = [lemmatizer.lemmatize(word) for word in document]
            document = [word.lower() for word in document]
            positional_indexer.add_document(doc_id, document)
            bigram_indexer.add_document(doc_id, document)
            doc_id += 1


def initialize():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    import_documents(file_address='Data/English.csv', language='en')
    print(positional_indexer.inverted_index)
    pass


def serve():
    pass
