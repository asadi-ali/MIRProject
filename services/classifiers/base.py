class BaseClassifier(object):

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        trains model on training data
        :param X: array-like or sparse matrix of shape (n_samples, n_features) 
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        :return:  
        """
        raise NotImplementedError()

    def predict(self, X):
        """
        predicts output of data
        :param X: array-like or sparse matrix of shape (n_samples, n_features) 
        :return: array-like of shape (n_samples,) or (n_samples, n_outputs)
        """
        raise NotImplementedError()

