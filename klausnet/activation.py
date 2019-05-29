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
        S = 1 / (1 + np.exp(-X))

        # 虽然S有normalize的作用，但是进来的X
        # 不知道为什么会变得非常非常大
        # 个人认为原因应该是由于跟新weight的时候
        # 导致这些数变得特别特别大

        # Calculate Gradients
        self.__gradient = np.multiply(S, (1 - S))
        return S

    @property
    def gradient(self):
        return self.__gradient

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
        self.__gradient = (exp_matrix * exp_matrix_sum - exp_matrix ** 2) / (exp_matrix_sum ** 2)
        output = exp_matrix / exp_matrix_sum
        # print(output.shape)
        return output

    @property
    def gradient(self):
        return self.__gradient


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
        self.__gradient = np.ones(X.shape)
        return X

    @property
    def gradient(self):
        return self.__gradient


class Relu(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.__gradient = (X >= 0).astype(int)
        output = np.multiply((X >= 0).astype(int), X)
        return output

    @property
    def gradient(self):
        return self.__gradient


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

