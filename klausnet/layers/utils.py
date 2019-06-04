from klausnet.layers import base
import numpy as np


# Dropout
class Dropout(base.Layer):
    def __init__(self, dropout_rate):
        '''

        :param dropout_rate:   % of params to drop
        '''
        super().__init__()
        self.dropout_rate = dropout_rate

    def train_forward(self, X):
        '''

        :param X:  无论x是什么dimension的
        :return:
        '''

        self.grad_x = np.ones(X.shape)

        self.shape_of_X = X.shape
        self.dimension = len(self.shape_of_X)

        self.total_param = 1
        for i in range(self.dimension):
            self.total_param *= self.shape_of_X[i]

        self.number_to_drop = self.total_param * self.dropout_rate

        self.drop_coord = np.zeros((self.number_to_drop, self.dimension))

        for i in range(self.dimension):
            self.drop_coord[:, i] = np.random.randint(0, self.shape_of_X[i] - 1, (self.number_to_drop, 1))

        for coord in self.drop_coord:
            X[coord] = 0
            self.grad_x[coord] = 0

        return X

    def pred_forward(self, X):
        return X

    @property
    def gradient(self):
        return {'back': self.grad_x}

# Batch Norm