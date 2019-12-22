import numpy as np
import sklearn.model_selection

from .base import BaseClassifier

NUMBER_OF_LABELS = 4


class KNNClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.K = None
        self.X_train = None
        self.y_train = None

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
                res[tag][1] = 1
            else:
                res[tag][0] = 1
        return res

    def calc_tag(self, data):
        distances = []
        for i in range(len(self.X_train)):
            distances.append((np.linalg.norm(self.X_train[i] - data), self.y_train[i]))

        distances.sort()
        tag_cnt = np.zeros(NUMBER_OF_LABELS)
        for i in range(min(len(distances), self.K)):
            tag_cnt[int(distances[i][1])] += 1

        return np.argmax(tag_cnt)

    def fit(self, X, y):
        y = self.convert_one_hot_to_tag(y)

        best_k = None
        best_score = -1000000000
        self.X_train, X_val, self.y_train, y_val = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
        for k in range(1, 22, 2):
            self.K = k
            score = 0
            for i in range(len(X_val)):
                score += self.calc_tag(X_val[i]) == y_val[i]

            if best_score < score:
                best_score = score
                best_k = k

        self.K = best_k
        self.X_train = X
        self.y_train = y

    def predict_proba(self, X):
        answer = np.zeros((len(X), NUMBER_OF_LABELS))
        for i in range(len(X)):
            answer[i] = self.get_one_hot(self.calc_tag(X[i]))
        return answer
