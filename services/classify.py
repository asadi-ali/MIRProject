import numpy as np
from .vectorspace import doc_to_vec
from sklearn.ensemble import RandomForestClassifier
from .document_manager import load_texts_and_tags, process_english_document,\
    documents_cnt, document_base, document_type, doc_indices_by_type
from .classifiers.naive_bayes import NaiveBayesClassifier
from .classifiers.svm import SVMClassifier
from .classifiers.knn import KNNClassifier
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict

clf = None
X_train = None
y_train = None
X_test = None
y_test = None

def load_train_data(path):
    global X_train, y_train
    X_train, y_train = load_labeled_data(path)

def load_test_data(path):
    global X_test, y_test
    X_test, y_test = load_labeled_data(path)

def train_classifier(classifier_type):
    global clf, X_train, y_train
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
    cnt = 0
    for text, tag in zip(texts, tags):
        one_hot = np.zeros(tags_cnt)
        one_hot[tag-1] = 1
        y.append(one_hot)
        X.append(doc_to_vec(process_english_document(text), documents_cnt()))
        cnt += 1
        # if cnt > 1000:
        #     break
    print("Data loading completed")
    return np.asarray(X), np.asarray(y)

def most_probable_label(prob):
    label = 0
    for i in range(len(prob)):
        if prob[i] > prob[label]:
            label = i
    return label

def classify_document(document):
    X = doc_to_vec(document, documents_cnt()).reshape(1, -1)
    p = clf.predict_proba(X)
    prob = [p[i][0][1] for i in range(len(p))]
    # print(prob)
    return most_probable_label(prob)


def classify_documents():
    print("Classifying documents")
    keys = doc_indices_by_type.keys()
    for key in keys:
        doc_indices_by_type[key] = []
    for id, document in enumerate(document_base):
        doc_indices_by_type[document_type[classify_document(document)]].append(id+1)
    print("Classification completed")

def report(X, y):
    p = clf.predict_proba(X)
    y_pred = []
    for j in range(len(X)):
        prob = [p[i][j][1] for i in range(len(p))]
        one_hot = np.zeros((len(p)))
        one_hot[most_probable_label(prob)] = 1
        y_pred.append(one_hot)
    y_pred = np.asarray(y_pred)
    print("accuracy: ", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred, target_names=document_type))

def report_on_train_data():
    global X_train, y_train
    report(X_train, y_train)

def report_on_test_data():
    global X_test, y_test
    report(X_test, y_test)

