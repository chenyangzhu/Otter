from otter import Variable
from otter.layers.common import Layer
import os
from tqdm import tqdm


class Model:
    def __init__(self):
        """
        In init, we input layers here.
        """
        pass

    def compile(self, optimizer, loss, epoch, batch):
        """
        :param graph:               The corresponding graph
        :param optimizer:           Optimizer
        :param loss:                Loss
        :param epoch:               # of iterations
        :param batch:               batch size, if = -1, then no batch
        :return:
        """
        self.optimizer = optimizer
        self.loss = loss
        self.epoch = epoch
        self.batch = batch
        self.compiled = True

    def add_saver(self, saver):
        """
        In this function, we create a saver account and save the metrics.
        Since savers are not mandatory for a model to work,
        so we would do not include the add saver function into the compile
        """
        self.saver = saver

    def forward(self, x):
        """
        This is a user-defined part, where user can define the forward
        function themselves.
        """
        pass

    def predict(self, x):
        return self.forward(x)

    def fit(self, x, y):
        # TODO This part is theoretically true
        #  but has not been tested so far
        iteration_number = int(x.shape[0] / self.batch)

        # An epoch is an iteration to the entire dataset
        for epoch_idx in (range(self.epoch)):
            print(f"Doing the {epoch_idx}th epoch.")

            for iteration_idx in tqdm(range(iteration_number)):
                # We need to split the dataset by using index
                start_idx = iteration_idx * self.batch
                end_idx = (iteration_idx + 1) * self.batch

                batch_x = Variable(x.value[start_idx:end_idx])
                batch_y = Variable(y.value[start_idx:end_idx])

                yhat = self.forward(batch_x)
                l = self.loss(batch_y, yhat)
                l.back_propagate_with_optimizer(self.optimizer)

    def save(self, path='./tmp'):

        # If the file does not exit, create this file
        if not os.path.isdir(path):
            os.mkdir(path)

        variables = self.__dict__
        number_of_layers = 0
        for each in variables.keys():
            # If layers[each] is a Layer object, we call and save it
            if issubclass(variables[each].__class__, Layer):
                variables[each].save_layer(path + '/layer'+str(number_of_layers)+'.json')
                number_of_layers += 1
        print(f"Model saved to {path}")

    def load(self, path='./tmp'):
        variables = self.__dict__
        number_of_layers = 0
        for each in variables.keys():
            # If layers[each] is a Layer object, we call and save it
            if issubclass(variables[each].__class__, Layer):
                variables[each].read_layer(path + '/layer'+str(number_of_layers)+'.json')
                number_of_layers += 1
        print(f"Model loaded from {path}")


class Sequential(Model):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.compiled = False

    def compile(self, optimizer, loss, epoch, batch):
        """
        :param graph:               The corresponding graph
        :param optimizer:           Optimizer
        :param loss:                Loss
        :param epoch:               # of iterations
        :param batch:               batch size, if = -1, then no batch
        :return:
        """

        self.optimizer = optimizer
        self.loss = loss
        self.epoch = epoch
        self.batch = batch
        self.compiled = True

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
                    X = each_layer.forward(X)

                output = self.loss(y=y, yhat=X)

                # Back-prop

                self.graph.back_propagate_with_optimizer(output, optimizer=self.optimizer)

                batch_loss += output.value
            # End batch
            hist_loss.append(batch_loss / int(self.n / self.batch))
        # End iteration

        history = {'loss': hist_loss}
        return history

    def predict(self, X):
        for each_layer in self.layers:

            X = each_layer.predict(X)
        return X