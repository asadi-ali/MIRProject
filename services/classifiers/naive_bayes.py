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
        res = np.zeros(NUMBER_OF_LABELS)
        res[tag] = 1
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
        answer = np.zeros((number_of_docs, NUMBER_OF_LABELS))
        for i in range(number_of_docs):
            prob = np.log(self.prior)
            for j in range(NUMBER_OF_LABELS):
                prob[j] += sum(np.log(self.likelihood[j] * X[i]))

            answer[i] = self.get_one_hot(np.argmax(prob))

        return answer




