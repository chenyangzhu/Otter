import numpy as np

import layers
from loss import mean_squared_error
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
                # print(X)

            # 运行最后的loss
            self.loss.forward(y=X, yhat=self.y)

            # Back-prop
            grad = self.loss.gradient  # Grad 是全局往前的 gradient
            for each_layer in reversed(self.layers):
                each_layer.update_model_gradient(grad)
                grad = np.matmul(each_layer.model_gradient, grad)

        # Do Gradient descent
            for each_layer in self.layers:
                self.optimizer.update_once(each_layer)

            hist_loss.append(self.loss.loss)


        history = {'loss':hist_loss}
        return history

if __name__ == "__main__":

    np.random.seed(2019)
    n = 10
    p = 100
    m = 10
    X = np.random.normal(0, 1, (n, p))
    w = np.random.normal(0, 1, (p, m))
    b = np.random.normal(0, 1, (n, m))
    y = np.matmul(X, w) + b

    model = Sequential([layers.Dense(hidden_unit=m,
                                     input_shape=(n, p),
                                     activation=Sigmoid()),
                        layers.Dense(hidden_unit=m,
                                     input_shape=(n, m),
                                     activation=Sigmoid())
                        ])

    model.compile(optimizer=GradientDescent(learning_rate=0.5),
                  loss=mean_squared_error(),
                  iteration=1000)
    history = model.fit(X, y)

    plt.plot(history['loss'])
    plt.show()




