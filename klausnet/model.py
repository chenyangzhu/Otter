import numpy as np

import layers
from loss import mean_squared_error, categorical_crossentropy
from optimizer import GradientDescent
from activation import Sigmoid
from nn import *

import matplotlib.pyplot as plt


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def compile(self, optimizer, loss, iteration):
        '''

        :param optimizer: Optimizer 类
        :param loss: Loss 类
        :param iteration:  # of iterations
        :return:
        '''
        self.optimizer = optimizer
        self.loss = loss
        self.iteration = iteration

    def fit(self, X, y):
        self.X = X  # self.X 永远是最初的X，而 X 随着后面不断改变。
        self.y = y

        # history
        hist_loss = []

        for i in range(self.iteration):

            X = self.X

            # 运行中间的layer
            for each_layer in self.layers:
                X = each_layer.forward(X)

            # 运行最后的loss
            self.loss.forward(y=X, yhat=self.y)

            # Back-prop
            grad = self.loss.gradient  # Grad 是全局往前的 gradient
            print(grad.shape)
            for each_layer in reversed(self.layers):
                each_layer.update_model_gradient(grad)
                # print(grad.shape)
                grad = np.matmul(grad, each_layer.model_gradient.T)

        # Do Gradient descent
            for each_layer in self.layers:
                self.optimizer.update_once(each_layer)

            hist_loss.append(self.loss.loss)
            print(self.loss.loss)

        history = {'loss':hist_loss}
        return history

    def predict(self, X):
        for each_layer in self.layers:
            X = each_layer.forward(X)
        return X


if __name__ == "__main__":
    np.random.seed(2013)

    n = 100
    p = 10
    m = 2
    X = np.random.normal(0, 1, (n, p))  # n x p
    w = np.random.normal(0, 1, (p, m))  # p x m
    b = np.random.normal(0, 1, (n, m))  # n x m
    y = np.sign(np.matmul(X, w) + b)    # n x m
    print(y)

    model = Sequential([layers.Dense(hidden_unit=128,
                                     input_shape=(n, p),
                                     activation=Sigmoid()),
                        layers.Dense(hidden_unit=m,
                                     input_shape=(n, 128),
                                     activation=Sigmoid())
                        ])

    model.compile(optimizer=GradientDescent(learning_rate=0.1),
                  loss=mean_squared_error(),
                  iteration=200)

    history = model.fit(X, y)

    yhat = model.predict(X)

    plt.plot(history['loss'])
    plt.show()

    # plt.plot(np.sum(yhat**2, axis=0))
    # plt.plot(np.sum(y**2, axis=0), c='r')
    # plt.show()
    # print(np.sum(yhat**2, axis=0))