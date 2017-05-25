import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from perceptron import Perceptron
from utilities import *
from AdalineGD import *


def perceptron_iris_train(X, y):
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('septal length')
    plt.ylabel('petal length')
    plt.legend(loc='upper left')
    plt.show()

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()

    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('sepal length[cm]')
    plt.ylabel('petal length[cm]')
    plt.legend(loc='upper left')
    plt.show()


def adaline_iris_train(X, y):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
    ada1 = AdalineGD(eta=0.01, n_iter=10).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate of 0.01')

    ax[1].plot(range(1, len(ada1.cost_) + 1), ada1.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate of 0.01')
    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[2].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('Sum-squared-error')
    ax[2].set_title('Adaline - Learning rate of 0.0001')

    plt.show()

def adaline_iris_train_std(X, y):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    X_std = np.copy(X)
    X_std[:, 0] = (X_std[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X_std[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    ax[0].scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    ax[0].scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    ax[1].scatter(X_std[:50, 0], X_std[:50, 1], color='red', marker='o', label='setosa')
    ax[1].scatter(X_std[50:100, 0], X_std[50:100, 1], color='blue', marker='x', label='versicolor')
    ax[0].set_title('Non-normalized')
    ax[1].set_title('Normalized')

    plt.show()

    ada = AdalineGD(n_iter=15, eta=0.01)
    ada.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada)

    plt.title('Adaline - Gradient descent')
    plt.xlabel('sepal length[normal]')
    plt.ylabel('petal length [normal]')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.show()

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

#perceptron_iris_train(X, y)
#adaline_iris_train(X, y)
adaline_iris_train_std(X, y)


