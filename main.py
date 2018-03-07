import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

class ANN(object):
    def __init__(self, M):
        self.M = M

    def fit(self, X, Y, learning_rate = 10e-6, reg = 10e-7, epochs = 1000, show_figure = False):
        X, Y = shuffle(X, Y)
        x_valid = X[-10:]
        y_valid = Y[-10:]
        t_valid = utils.y2indicator(y_valid)

        x = X[:-10]
        y = Y[:-10]
        t = utils.y2indicator(y)

        N, D = x.shape
        K = len(set(y))

        self.W1 = np.random.randn(D, self.M)
        self.b1 = np.random.randn(self.M)

        self.W2 = np.random.randn(self.M, K)
        self.b2 = np.random.randn(K)

        costs = []

        for i in range(epochs):
            pY, Z = self.forward(x)

            #Updating Weights
            D = pY - t
            self.W2 -= learning_rate * (Z.T.dot(D) + reg * self.W2)
            self.b2 -= learning_rate * (D.sum() + reg * self.b2)

            dZ = D.dot(self.W2.T) * Z * (1 - Z)
            self.W1 -= learning_rate * (x.T.dot(dZ) + reg * self.W1)
            self.b1 -= learning_rate * (dZ.sum() + reg * self.b1)

            if i % 10 == 0:
                pY_valid, _= self.forward(x_valid)
                c = utils.cost(t_valid, pY_valid)
                costs.append(c)
                e = utils.error_rate(y_valid, np.argmax(pY_valid, axis = 1))
                print("i:", i, " cost: ", c, " error: ", e)

        if show_figure:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = utils.sigmoid(X.dot(self.W1) + self.b1)
        Y = utils.softmax(Z.dot(self.W2) + self.b2)
        return Y, Z

    def predict(self, X):
        pY, _ = self.forward(X)
        return pY

    def score(self, X, Y):
        prediction = self.predict(X)
        return utils.error_rate(Y, prediction)

def main():
    X, Y = utils.get_data()
    Y = utils.encode_labels(Y)
    print(Y[-10:])
    model = ANN(200)
    model.fit(X, Y, reg = 0, show_figure=True)


main()