# based on this notebook: https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras

import numpy as np
import pandas as pd

# Preparing data
data = pd.read_csv("mnist.csv")
data = np.array(data)

m, n = data.shape
np.random.shuffle(data)  # shuffle before splitting into dev and training sets

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.0
_, m_train = X_train.shape

data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.0


# Classes & Functions
class MLP:
    def __init__(self, structure):
        # structure of nn
        self.structure = structure
        self.Z = []
        self.A = []
        self.dW = []
        self.db = []

        # weights & biases
        self.weights = [
            np.random.rand(self.structure[i + 1], self.structure[i]) - 0.5
            for i in range(len(self.structure) - 1)
        ]
        self.biases = [
            np.random.rand(self.structure[i + 1], 1) - 0.5
            for i in range(len(self.structure) - 1)
        ]

    def _relu(self, Z):
        return np.maximum(Z, 0)

    # derivative of relu
    # f(x)=x -> f'(x)=1
    def _drelu(self, Z):
        # true = 1, false = 0
        return Z > 0

    def _softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def _one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def _forward(self, X):
        for i in range(len(self.structure) - 1):
            Z = self.weights[i].dot(X) + self.biases[i]
            self.Z.append(Z)
            # last layer do softmax else do activation function
            if i == len(self.structure) - 2:
                A = self._softmax(Z)
                self.A.append(A)
            else:
                X = self._relu(Z)
                self.A.append(X)

    def _backprop(self, X, Y):
        layer = len(self.Z)
        oh_Y = self._one_hot(Y)
        for i in range(layer):
            # output layer
            if i == 0:
                dZ = self.A[layer - 1] - oh_Y
                dW = 1 / m * dZ.dot(self.A[layer - 2].T)
                db = 1 / m * np.sum(dZ)
                self.dW.append(dW)
                self.db.append(db)
            # input layer
            elif i == layer - 1:
                dZ = self.weights[layer - i].T.dot(dZ) * self._drelu(
                    self.Z[layer - i - 1]
                )
                dW = 1 / m * dZ.dot(X.T)
                db = 1 / m * np.sum(dZ)
                self.dW.append(dW)
                self.db.append(db)
            # hidden layer
            else:
                dZ = self.weights[layer - i].T.dot(dZ) * self._drelu(
                    self.Z[layer - i - 1]
                )
                dW = 1 / m * dZ.dot(self.A[layer - i - 1].T)
                db = 1 / m * np.sum(dZ)
                self.dW.append(dW)
                self.db.append(db)
                pass

        # make the elements match to forward pass
        # (we appended them is reverse order because we went through the layers backward)
        self.dW.reverse()
        self.db.reverse()

    def _update_params(self, learn_rate):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - learn_rate * self.dW[i]
            self.biases[i] = self.biases[i] - learn_rate * self.db[i]

    def _clear(self):
        self.Z = []
        self.A = []
        self.dW = []
        self.db = []

    def _get_prediction(self):
        return np.argmax(self.A[-1], 0)

    def _show_accuracy(self, Y):
        # get predictions
        pred = self._get_prediction()
        print("Prediction: ", pred, "Real Value: ", Y)
        print("Accuracy: ", np.sum(pred == Y) / Y.size)

    def train(self, X, Y, learn_rate, iterations):
        m, _ = X_train.shape
        if m != self.structure[0]:
            raise Exception(
                f"Shape of training data ({m}) doesn't match shape of input layer ({self.structure[0]})'"
            )

        if Y_train.ndim != 1:
            raise Exception("The true values (ground truth) must be in a 1D-list.")

        for i in range(iterations):
            self._forward(X)
            self._backprop(X, Y)
            self._update_params(learn_rate)

            if i % 50 == 0:
                print("\nIteration: ", i)
                self._show_accuracy(Y)

            # clear derivatives, activations, Z for next epoch
            self._clear()

    def predict(self, X, Y):
        for i in range(100):
            self._forward(X[:, i, None])
            pred = self._get_prediction()
            print("GT: ", Y[i], " Prediction: ", pred)


mlp = MLP([784, 20, 20, 10])
mlp.train(X_train, Y_train, 0.1, 501)
mlp.predict(X_test, Y_test)
