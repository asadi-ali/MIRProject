import numpy as np
import sklearn.svm
import sklearn.model_selection

from .base import BaseClassifier

NUMBER_OF_LABELS = 4


class SVMClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.model = None

    @staticmethod
    def convert_one_hot_to_tag(y):
        answer = np.zeros(len(y))
        for i in range(len(y)):
            answer[i] = np.argmax(y[i])
        return answer

    @staticmethod
    def get_one_hot(tag):
        res = np.zeros((NUMBER_OF_LABELS, 2))
        for i in range(NUMBER_OF_LABELS):
            if i == tag:
                res[i][1] = 1
            else:
                res[i][0] = 1
        return res

    def fit(self, X, y):
        y = self.convert_one_hot_to_tag(y)

        best_score = -1000000000
        for C in [0.5, 1, 1.5, 2]:
            print("set C = ", C)
            X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
            svc = sklearn.svm.SVC(C=C, gamma='scale', decision_function_shape='ovo')
            svc.fit(X_train, y_train)
            score_sum = svc.score(X_val, y_val)
            if score_sum > best_score:
                best_score = score_sum
                self.model = svc

    def predict_proba(self, X):
        res = self.model.predict(X)
        answer = np.zeros((NUMBER_OF_LABELS, len(X), 2))
        for j in range(len(X)):
            one_hot = self.get_one_hot(res[j])
            for i in range(NUMBER_OF_LABELS):
                answer[i][j] = one_hot[i]
        return answer
