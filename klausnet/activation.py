"""
Activation Functions
- Sigmoid
- Softmax
- Tanh
- Linear
- Relu
"""


import numpy as np


class Activation:
    def __init__(self):
        pass

    def forward(self, X):
        return X


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


# TODO Write Gradients
class Softmax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        '''
        :param X: n x p matrix
        :return:  n x p matrix
        '''
        # Forward
        exp_matrix = np.exp(X)
        exp_matrix_sum = np.sum(exp_matrix, axis=1).reshape(exp_matrix.shape[0],1)

        # Backward
        # print(exp_matrix.shape)
        # print(exp_matrix_sum.shape)
        self.gradient = (exp_matrix * exp_matrix_sum - exp_matrix ** 2) / (exp_matrix_sum ** 2)
        output = exp_matrix / exp_matrix_sum
        print(output.shape)
        return output


class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.X = X
        return np.tanh(X)

    @property
    def gradient(self):
        return 1 - np.tanh(self.X) ** 2


class Linear(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.X = X
        return X

    @property
    def gradient(self):
        return np.ones(self.X.shape)


class Relu(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.X = X
        return np.multiply((X >= 0).astype(int), X)

    @property
    def gradient(self):
        return (self.X >= 0).astype(int)


# class LeakyRelu(Activation):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, X):
#         self.X = X
#         return np.multiply((X >= 0).astype(int), X)
#
#     @property
#     def gradient(self):
#         return (self.X >= 0).astype(int)

