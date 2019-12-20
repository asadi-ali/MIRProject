import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def get_random_forest_classifier(X_train, y_train):
    clf = RandomForestClassifier() # TODO
    clf.fit(X_train, y_train)
    return clf


def load_data(path):
    """
    :param path: path to data  
    :return: X, yg as numpy arrays
    """
    data = pd.read_csv(path)
    print(data)


X_train, y_train = load_data('../data/phase2_train.csv')
X_test, y_test = load_data('../data/phase2_test.csv')
random_forest_classifier = get_random_forest_classifier(X_train, y_train)
