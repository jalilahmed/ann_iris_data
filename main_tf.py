import utils
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

class ANN_TF(object):
    def __init__(self, M):
        self.M = M

    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape,stddev= 0.01))


    def fit(self, X, Y, learning_rate = 0.01, epochs = 1000, show_figure = False):
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        X_valid = X[-10:]
        Y_valid = Y[-10:]
        T_valid = utils.y2indicator(Y_valid)

        X = X[:-10]
        Y = Y[:-10]
        T = utils.y2indicator(Y)

        N, D = X.shape
        K = len(set(Y))

        tfX = tf.placeholder(tf.float32, [None, D])
        tfY = tf.placeholder(tf.float32, [None, K])

        self.W1 = self.init_weights([D, self.M])
        self.b1 = self.init_weights([self.M])

        self.W2 = self.init_weights([self.M, K])
        self.b2 = self.init_weights([K])

        py_x = self.forward(X)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tfY, logits=py_x))
        tf.summary.scalar('cost', cost)

        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        predict_op = tf.argmax(py_x, 1)

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(epochs):
            sess.run(train_op, feed_dict = {tfX: X , tfY: T })
            prediction = sess.run(predict_op, feed_dict = {tfX: X_valid, tfY: T_valid})
            if i % 10 == 0:
                print("i: ", i, "accuracy: ", np.mean(Y == prediction))


    def forward(self, X):
        Z = tf.nn.sigmoid(tf.matmul(X, self.W1) + self.b1)
        Y = tf.matmul(Z, self.W2) + self.b2
        return Y

def main():
    X, Y = utils.get_data()
    Y = utils.encode_labels(Y)
    model = ANN_TF(200)
    model.fit(X, Y)

if __name__ == "__main__":
    main()