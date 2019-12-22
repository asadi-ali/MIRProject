import numpy as np
from .vectorspace import doc_to_vec
from sklearn.ensemble import RandomForestClassifier
from .document_manager import load_texts_and_tags, process_english_document,\
    documents_cnt, document_base, doc_indices_by_type, document_type
from .classifiers.naive_bayes import NaiveBayesClassifier
from .classifiers.svm import SVMClassifier
from .classifiers.knn import KNNClassifier

clf = None


def train_classifier(X_train, y_train, classifier_type):
    global clf
    if classifier_type == 'random_forest':
        print("Training random forest on training data")
        clf = RandomForestClassifier(n_estimators=20)
        clf.fit(X_train, y_train)
        print("Training completed")
    elif classifier_type == 'naive_bayes':
        print("Training naive_bayes on training data")
        clf = NaiveBayesClassifier()
        clf.fit(X_train, y_train)
        print("Training completed")
    elif classifier_type == 'svm':
        print("Training svm on training data")
        clf = SVMClassifier()
        clf.fit(X_train, y_train)
        print("Training completed")
    else:
        print("Training knn on training data")
        clf = KNNClassifier()
        clf.fit(X_train, y_train)
        print("Training completed")


def load_labeled_data(path):
    print("Loading data from ", path)
    """
    :param path: data file address 
    :return: X, y data and its label
    """
    texts, tags = load_texts_and_tags(path)
    X = []
    y = []
    tags_cnt = max(tags)
    for text, tag in zip(texts, tags):
        one_hot = np.zeros(tags_cnt)
        one_hot[tag-1] = 1
        y.append(one_hot)
        X.append(doc_to_vec(process_english_document(text), documents_cnt()))
    print("Data loading completed")
    return np.asarray(X), np.asarray(y)


def classify_document(document):
    X = [doc_to_vec(document, documents_cnt())]
    y = clf.predict(X)[0]
    print(y)
    for i in range(len(y)):
        if y[i] > 0:
            return i
    return 0


def classify_documents():
    for id, document in enumerate(document_base):
        doc_indices_by_type[document_type[classify_document(document)]].append(id+1)
