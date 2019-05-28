import numpy as np
from klausnet.history import history_recorder

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

    def record(self, recorder_list):
        self.recorder = history_recorder(recorder_list)

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
            self.loss.forward(y=self.y, yhat=X)

            # Back-prop
            grad = self.loss.gradient  # Grad 是全局往前的 gradient
            # print(grad.shape)
            for each_layer in reversed(self.layers):
                each_layer.update_model_gradient(grad)
                # print(grad.shape)
                grad = np.matmul(grad, each_layer.model_gradient.T)


            # Update Weights
            for each_layer in self.layers:
                self.optimizer.update_once(each_layer)


            hist_loss.append(self.loss.loss)
            # print(self.loss.loss)

        history = {'loss': hist_loss}
        return history

    def predict(self, X):
        for each_layer in self.layers:
            X = each_layer.forward(X)
        return X
