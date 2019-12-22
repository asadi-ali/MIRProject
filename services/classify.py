import pandas as pd
import numpy as np
from .vectorspace import doc_to_vec
from sklearn.ensemble import RandomForestClassifier
from .document_manager import load_texts_and_tags, process_english_document
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

def get_trained_classifier(X_train, y_train, classifier_type):
    if classifier_type == 'random_forest':
        print("Training random forest on training data")
        clf = RandomForestClassifier(n_estimators=20)
        clf.fit(X_train, y_train)
        print("Training completed")
        return clf
    else:
        raise NotImplementedError

def one_hot(tag, tags_cnt):
    res = np.zeros(tags_cnt)
    res[tag] = 1
    return res

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
        y.append(one_hot(tag-1, tags_cnt))
        X.append(doc_to_vec(process_english_document(text), {'tf':'n', 'idf':'t', 'norm':'n'}))
    print("Data loading completed")
    return np.asarray(X), np.asarray(y)

def test_classifier():
    pass