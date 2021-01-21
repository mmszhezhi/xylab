import math
import numpy as np


class SGDL2RegularizedLinearModel(object):

    def __init__(self, n_epochs, learning_rate, batch_size, regularization_strength=0.0):
        self._w = None
        self._n_epochs = n_epochs
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._regularization_strength = regularization_strength

    def gradient(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # x_0 is always 1
        self._w = np.random.randn(X.shape[1], 1)
        batch_size = self._batch_size if self._batch_size is not None else X.shape[0]
        for i in range(self._n_epochs):
            for j in range(X.shape[0] / batch_size):
                learning_rate = self._learning_rate if isinstance(self._learning_rate, float) \
                    else self._learning_rate(i * (X.shape[0] / batch_size) + j)
                sample = np.random.choice(X.shape[0], batch_size, replace=False)
                self._w -= learning_rate * self.gradient(X[sample, :], y[sample])


class SGDLinearRegression(SGDL2RegularizedLinearModel):

    def gradient(self, X, y):
        gradient = np.zeros((X.shape[1], 1))
        for xi, yi in zip(X, y):
            gradient += np.reshape((np.dot(np.transpose(self._w), xi) - yi) * xi, (X.shape[1], 1))
        return gradient * (2.0 / X.shape[0])

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # x_0 is always 1
        return np.dot(np.transpose(self._w), np.transpose(X)).flatten()


class SGDRidgeRegression(SGDL2RegularizedLinearModel):

    def gradient(self, X, y):
        gradient = np.zeros((X.shape[1], 1))
        for xi, yi in zip(X, y):
            gradient += np.reshape((np.dot(np.transpose(self._w), xi) - yi) * xi, (X.shape[1], 1))
        gradient *= (2.0 / X.shape[0])
        return gradient + 2.0 * self._regularization_strength * self._w

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # x_0 is always 1
        return np.dot(np.transpose(self._w), np.transpose(X)).flatten()


class SGDLogisticRegression(SGDL2RegularizedLinearModel):

    def theta(self, s):
        return (math.e ** s) / (1 + math.e ** s)

    def gradient(self, X, y):
        gradient = np.zeros((X.shape[1], 1))
        for xi, yi in zip(X, y):
            gradient += np.reshape(self.theta(-yi * np.dot(np.transpose(self._w), xi)) * yi * xi,
                                   (X.shape[1], 1))
        return gradient * (-1.0 / X.shape[0])

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # x_0 is always 1
        return np.vectorize(lambda x: self.theta(x))(
            np.dot(np.transpose(self._w), np.transpose(X)).flatten()
        )


class SGDL2RegularizedLogisticRegression(SGDL2RegularizedLinearModel):

    def theta(self, s):
        return (math.e ** s) / (1 + math.e ** s)

    def gradient(self, X, y):
        gradient = np.zeros((X.shape[1], 1))
        for xi, yi in zip(X, y):
            gradient += np.reshape(self.theta(-yi * np.dot(np.transpose(self._w), xi)) * yi * xi, (X.shape[1], 1))
        gradient *= (-1.0 / X.shape[0])
        return gradient + 2.0 * self._regularization_strength * self._w

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # x_0 is always 1
        return np.vectorize(lambda x: self.theta(x))(
            np.dot(np.transpose(self._w), np.transpose(X)).flatten()
        )
