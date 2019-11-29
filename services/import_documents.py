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

doc_id = 1

def get_stopwords(document):
    counter = Counter(document)
    frac = 75
    stop_num = min(100, len(counter)//frac)
    stopwords = [rec[0] for rec in counter.most_common(stop_num)]
    return stopwords


def import_document(text, tokenizer, union, normalizer=None, stemmer=None, lemmatizer=None, stopwords=None):
    #print(text)
    global doc_id
    if normalizer is not None:
        text = normalizer.normalize(text)
    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    document = tokenizer.tokenize(text)
    union.extend(document)
    if stopwords is None:
        stopwords = get_stopwords(document)
    document = [word for word in document if not word in stopwords]
    if stemmer is not None:
        document = [stemmer.stem(word) for word in document]
    if lemmatizer is not None:
        document = [lemmatizer.lemmatize(word) for word in document]
    document = [word.lower() for word in document]
    print(document)
    #get_stopwords(document)
    positional_indexer.add_document(doc_id, document)
    bigram_indexer.add_document(doc_id, document)
    doc_id += 1


def import_english_documents(file_address):
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()
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
    en_tokens = []
    for text in raw_documents:
        import_document(text, tokenizer, en_tokens, stemmer=stemmer, stopwords=stop_words)
    print("English most common words:")
    print(get_stopwords(en_tokens))

def import_persian_documents(file_address):
    global doc_id
    with open(file_address, 'rt') as xmlfile:
        docs = xmlfile.read()
    titles = regex.findall('(?<=<title>).*(?=<\/title>)', docs)
    texts = regex.findall('(?<=<text xml:space="preserve">).*?(?=<\/text>)', docs, flags=regex.DOTALL)
    normalizer = hazm.Normalizer()
    tokenizer = hazm.WordTokenizer()
    stemmer = hazm.Stemmer()
    lemmatizer = hazm.Lemmatizer()
    raw_documents = [title + ' ' + text for title, text in zip(titles, texts)]
    raw_documents = [text.translate(str.maketrans('', '', string.ascii_letters)) for text in raw_documents]
    fa_tokens = []
    for text in raw_documents:
        import_document(text, tokenizer, fa_tokens, normalizer=normalizer, stemmer=stemmer)
    print("Persian   most common words:")
    print(get_stopwords(fa_tokens))

def import_documents():
    import_english_documents('data/English.csv')
    import_persian_documents('data/Persian.xml')
