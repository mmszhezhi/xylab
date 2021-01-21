import math
import numpy as np
import numpy
from sklearn.linear_model import LinearRegression

class SolvePolynomial():
    def __init__(self, order):
        self.order = order
        self.model = None

    def monomial(self, a, b):
        return lambda x: a * math.pow(x, b)

    def polyList(self, order):
        return [self.monomial(1, i) for i in range(0, order + 1)]

    def evaluate(self, functionList, x):
        return sum([f(x) for f in functionList])

    def weightedSum(self, w, F):
        if (len(w) != len(F)):
            raise Exception("Function/weight size mismatch")
        else:
            return lambda x: sum([w[i] * F[i](x) for i in range(0, len(w))])

    def gradient(self, w, x, y):
        y_estimate = x.dot(w).flatten()
        error = (y.flatten() - y_estimate)
        gradient = -(1.0 / len(x)) * error.dot(x)
        return gradient, np.pow(error, 2)

    def polyTrain(self, x):
        F = self.polyList(self.order)
        design = [[f(i) for f in F] for i in x.index.array]
        design = numpy.asarray(design)
        pinv = numpy.linalg.pinv(design)
        t = numpy.asarray(x.array).T
        w = numpy.dot(pinv, t)
        self.predict = self.weightedSum(w, F)
        return len(x.index)

    def compute_cost(self, X, y, theta):
        Hx = np.dot(X, theta)
        return np.dot((Hx - y).transpose(), Hx - y) / (2 * X.shape[0])

    def gradient_descent(self, X, y, theta, learning_rate, iteration):

        cost_his = np.zeros((iteration, 1))

        cost_his[0] = self.compute_cost(X, y, theta)

        for i in range(1, iteration):
            Hx = np.dot(X, theta)

            theta = theta - (learning_rate / X.shape[0]) * (np.dot((Hx - y).transpose(), X)).transpose()
            # temp = theta[0, 0] - (alpha / Data.shape[0] * np.dot(Data[:, :1].T, (np.dot(Data, theta.T) - Y)) ).T
            cost_his[i] = self.compute_cost(X, y, theta)

        return theta, cost_his

    def feature_normalize(self, X):

        mean = np.mean(X, axis=0, keepdims=True)

        std = np.std(X, axis=0, keepdims=True)

        X_norm = (X - mean) / std

        return X_norm

    def normal_equation(self, X, y):

        a = np.dot(X.transpose(), X)

        b = np.dot(X.transpose(), y)

        return np.dot(np.linalg.inv(a), b)

    def trainlr(self):
        if not self.model:
            self.model = LinearRegression()


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

    def CostFunction(self,X,Y):
        h = np.dot(X, self._w)
        cost = 1 / float((2 * X.shape[0])) * (np.sum((h - Y) ** 2) + np.sum(self._w ** 2))
        return cost

    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # x_0 is always 1
        self._w = np.random.randn(X.shape[1], 1)
        batch_size = self._batch_size if self._batch_size is not None else X.shape[0]
        for i in range(self._n_epochs):
            for j in range(math.ceil(X.shape[0] / batch_size)):
                learning_rate = self._learning_rate if isinstance(self._learning_rate, float) \
                    else self._learning_rate(i * (X.shape[0] / batch_size) + j)
                sample = np.random.choice(X.shape[0], batch_size, replace=False)
                self._w -= learning_rate * self.gradient(X[sample, :], y[sample])
                print(self.CostFunction(X[sample, :], y[sample]),self._w)


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






