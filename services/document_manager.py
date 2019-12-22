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
from collections import Counter
import pandas as pd
from collections import defaultdict

en_stop_words = set(stopwords.words('english'))
en_tokenizer = RegexpTokenizer(r'\w+')
en_stemmer = PorterStemmer()
en_lemmatizer = nltk.WordNetLemmatizer()
fa_normalizer = hazm.Normalizer()
fa_tokenizer = hazm.WordTokenizer()
fa_stemmer = hazm.Stemmer()
fa_lemmatizer = hazm.Lemmatizer()


doc_id = 1
document_base = []
raw_document_base = []
en_common = []
fa_common = []
en_tokens = []
fa_tokens = []
doc_indices_by_type = defaultdict(list)
document_type = ['World', 'Sports', 'Business', 'Sci/Tech']

def documents_cnt():
    return len(document_base)

def get_stopwords(document):
    counter = Counter(document)
    frac = 75
    stop_num = min(100, len(counter)//frac)
    stopwords = [rec[0] for rec in counter.most_common(stop_num)]
    return stopwords

def process_document(text, tokenizer, union=None, normalizer=None, stemmer=None, lemmatizer=None, stopwords=None):
    if normalizer is not None:
        text = normalizer.normalize(text)
    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    document = tokenizer.tokenize(text)
    if union is not None:
        union.extend(document)
    if stopwords is None:
        stopwords = get_stopwords(document)
    document = [word for word in document if not word in stopwords]
    if stemmer is not None:
        document = [stemmer.stem(word) for word in document]
    if lemmatizer is not None:
        document = [lemmatizer.lemmatize(word) for word in document]
    document = [word.lower() for word in document]
    document = [word for word in document if word]
    return document

def process_english_document(text, union=None):
    return process_document(text, en_tokenizer, union, normalizer=None, stemmer=en_stemmer, lemmatizer=en_lemmatizer, stopwords=en_stop_words)

def process_farsi_document(text, union=None):
    return process_document(text, fa_tokenizer, union, normalizer=fa_normalizer, stemmer=fa_stemmer)

def import_document(document):
    global doc_id
    document_base.append(document)
    positional_indexer.add_document(doc_id, document)
    bigram_indexer.add_document(doc_id, document)
    doc_id += 1

def import_english_document(text):
    raw_document_base.append(text)
    import_document(process_english_document(text, en_tokens))

def import_farsi_document(text):
    raw_document_base.append(text)
    import_document(process_farsi_document(text, fa_tokens))

def import_english_documents(file_address):
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # nltk.download('wordnet')
    raw_documents = []
    with open(file_address, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        title = True
        for row in reader:
            if title:
                title = False
                continue
            text = ' '.join(row, )
            raw_documents.append(text)
    for text in raw_documents:
        import_english_document(text)
    global en_common
    en_common = get_stopwords(en_tokens)

def import_farsi_documents(file_address):
    global doc_id
    with open(file_address, 'rt') as xmlfile:
        docs = xmlfile.read()
    titles = regex.findall('(?<=<title>).*(?=<\/title>)', docs)
    texts = regex.findall('(?<=<text xml:space="preserve">).*?(?=<\/text>)', docs, flags=regex.DOTALL)
    raw_documents = [title + ' ' + text for title, text in zip(titles, texts)]
    raw_documents = [text.translate(str.maketrans('', '', string.ascii_letters)) for text in raw_documents]
    for text in raw_documents:
        import_farsi_document(text)
    global fa_common
    fa_common = get_stopwords(fa_tokens)


def import_documents():
    import_english_documents('data/English.csv')
    # import_farsi_documents('data/Persian.xml')


def get_farsi_commons():
    return fa_common

def get_english_commons():
    return en_common

def load_texts_and_tags(path):
    """
    :param path: path to data  
    :return: x, y as texts, tags arrays
    """
    data = pd.read_csv(path)
    tags = data['Tag']
    texts = data['Title'] + ' ' + data['Text']
    return texts, tags