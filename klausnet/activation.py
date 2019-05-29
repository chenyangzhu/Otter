"""
Activation Functions
- Sigmoid
- Softmax
- Tanh
- Linear
- Relu

注意了，所有的
@property
    def gradient(self):
        return {'X': self.__gradient}

return dict 的目的是Activation 要和 Layer 相统一
本质上 Activation 应该是和Layer一样的东西。
"""


import numpy as np
from klausnet.layers import Layer


class Activation(Layer):
    def __init__(self):
        super().__init__()
        pass


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
        return {'X': self.__gradient}

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

        # Back-prop
        self.__gradient = (exp_matrix * exp_matrix_sum - exp_matrix ** 2) / (exp_matrix_sum ** 2)
        output = exp_matrix / exp_matrix_sum
        return output

    @property
    def gradient(self):
        return {'X': self.__gradient}


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
        return {'X': self.__gradient}


class Relu(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.__gradient = (X >= 0).astype(int)
        output = np.multiply((X >= 0).astype(int), X)
        return output

    @property
    def gradient(self):
        return {'X': self.__gradient}


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
