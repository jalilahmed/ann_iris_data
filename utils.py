import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def get_data():
    X = []
    Y = []
    first = True
    for line in open('iris.csv'):
        if first:
            first = False
        else:
            Y.append(line.split(',')[-1])
            row = line.split(',')[:-1]
            temp = []
            for i in range(len(row)):
                temp.append(float(row[i]))
            X.append(temp)
    X = np.array(X)
    X = (X - X.mean(axis = 0)) / X.std(axis = 0)
    return X, Y


def encode_labels(Y):
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_y = encoder.transform(Y)
    return encoded_y

def y2indicator(Y):
    N = len(Y)
    K = len(set(Y))
    T = np.zeros((N, K))
    for i in range(N):
        T[i, int(Y[i])] = 1
    return T

def cost(T, Y):
    return -(T * np.log(Y)).sum()

def error_rate(T, Y):
    return np.mean(T != Y)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(a):
    exp_a = np.exp(a)
    return exp_a / exp_a.sum(axis =1 , keepdims = True)

def accuracy(T, Y):
    return np.mean(T == Y)


