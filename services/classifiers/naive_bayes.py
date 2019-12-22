from .base import BaseClassifier
import numpy as np

NUMBER_OF_LABELS = 4


class NaiveBayesClassifier(BaseClassifier):

    def __init__(self):
        super().__init__()
        self.prior = None
        self.likelihood = None

    @staticmethod
    def get_tag(one_hot):
        for i, x in enumerate(one_hot):
            if x:
                return i
        return -1

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
        number_of_docs = len(X)
        number_of_tokens = len(X[0])

        self.prior = np.zeros(NUMBER_OF_LABELS)
        self.likelihood = np.zeros((NUMBER_OF_LABELS, number_of_tokens))

        for i in range(number_of_docs):
            self.prior += y[i] / number_of_docs
            for j in range(number_of_tokens):
                self.likelihood[self.get_tag(y[i])][j] += X[i][j]

        for i in range(NUMBER_OF_LABELS):
            self.likelihood[i] /= sum(self.likelihood[i])

        print(self.likelihood)
        print(self.prior)

    def predict_proba(self, X):
        number_of_docs = len(X)
        answer = np.zeros((NUMBER_OF_LABELS, number_of_docs, 2))
        for j in range(number_of_docs):
            prob = np.log(self.prior)
            for i in range(NUMBER_OF_LABELS):
                prob[i] += sum(np.log(self.likelihood[i] * X[j]))

            one_hot = self.get_one_hot(np.argmax(prob))
            for i in range(NUMBER_OF_LABELS):
                answer[i][j] = one_hot[i]

        return answer




