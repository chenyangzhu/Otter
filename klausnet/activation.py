import numpy as np

class Activation:
    def __init__(self):
        pass

    def forward(self, X):
        return X

    @property
    def gradient(self):
        return 0


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        # self.X = X
        self.S = 1 / (1 + np.exp(-X))
        return self.S

    @property
    def gradient(self):
        return np.multiply(self.S, (1 - self.S))

class Softmax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        '''

        :param X: n x p matrix
        :return: n x p matrix
        '''
        exp_matrix = np.exp(X)
        exp_matrix_sum = np.sum(exp_matrix, axis = 0)
        return exp_matrix / exp_matrix_sum

    @property
    def gradient(self):
        return