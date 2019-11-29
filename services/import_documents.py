from __future__ import unicode_literals
import csv
import regex
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import hazm
from services.index import positional_indexer, bigram_indexer

doc_id = 1

def import_english_documents(file_address):
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # nltk.download('wordnet')
    global doc_id
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()
    with open(file_address, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        title = True
        for row in reader:
            if title:
                title = False
                continue
            text = ' '.join(row, )
            document = tokenizer.tokenize(text)
            document = [word for word in document if not word in stop_words]
            document = [stemmer.stem(word) for word in document]
            document = [lemmatizer.lemmatize(word) for word in document]
            document = [word.lower() for word in document]
            positional_indexer.add_document(doc_id, document)
            bigram_indexer.add_document(doc_id, document)
            doc_id += 1

def import_persian_documents(file_address):
    global doc_id
    with open(file_address, 'rt') as xmlfile:
        docs = xmlfile.read()
    titles = regex.findall('(?<=<title>).*(?=</title>)', docs)
    texts = regex.findall('(?<=<text.*?>).*?(?=<\/text>)', docs, flags=regex.DOTALL)
    texts = [text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) for text in texts]
    titles = [title.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) for title in titles]
    normalizer = hazm.Normalizer()
    tokenizer = hazm.WordTokenizer()
    stemmer = hazm.Stemmer()
    lemmatizer = hazm.Lemmatizer()
    for title, text in zip(titles, texts):
        text = title + ' ' + text
        document = normalizer.normalize(text)
        document = tokenizer.tokenize(document)
        document = [stemmer.stem(word) for word in document]
        document = [lemmatizer.lemmatize(word) for word in document]
        positional_indexer.add_document(doc_id, document)
        bigram_indexer.add_document(doc_id, document)
        doc_id += 1