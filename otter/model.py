from otter.history import history_recorder
from otter import Variable
import numpy as np
from otter.optimizer import StochasticGradientDescent


class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.compiled = False

    def compile(self, graph, optimizer, loss, epoch, batch):
        """
        :param graph:               The corresponding graph
        :param optimizer:           Optimizer
        :param loss:                Loss
        :param epoch:               # of iterations
        :param batch:               batch size, if = -1, then no batch
        :return:
        """

        self.graph = graph
        self.optimizer = optimizer
        self.loss = loss
        self.epoch = epoch
        self.batch = batch
        self.compiled = True

    def record(self, recorder_list):
        self.recorder = history_recorder(recorder_list)

    def fit(self, X, y):
        # check if compiled
        if not self.compiled:
            raise AttributeError("You need to compile the model first.")

        self.X = X
        self.y = y

        self.n = X.shape[0]

        # history
        hist_loss = []

        for i in range(self.epoch):

            if i % 10 == 0:
                print("The", i, "th epoch.")

            batch_loss = 0

            # Start batch
            for j in range(int(self.n / self.batch)):

                # Select Batch Data
                X = Variable(self.X.value[j * self.batch: (j+1) * self.batch])
                y = Variable(self.y.value[j * self.batch: (j+1) * self.batch])

                # run layers
                for each_layer in self.layers:
                    X = each_layer.train_forward(X)

                output = self.loss(y=y, yhat=X)

                # Back-prop

                self.graph.update_gradient_with_optimizer(output, optimizer=self.optimizer)

                batch_loss += output.value
            # End batch
            hist_loss.append(batch_loss / int(self.n / self.batch))
        # End iteration

        history = {'loss': hist_loss}
        return history

    def predict(self, X):
        for each_layer in self.layers:
            X = each_layer.predict_forward(X)
        return X