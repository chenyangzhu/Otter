from klausnet.layers import base
import numpy as np


class Dense(base.Layer):
    def __init__(self, hidden_unit, input_shape, activation, learnable=True):

        '''
        :param hidden_unit:   # of hidden units
        :param input_shape:   input shape
        :param activation:    Activation Class
        :param learnable:     learnable=True，则在gradient descent 的时候更新这层的param，如果False，则不更新
        '''

        super().__init__()
        # Normal initialization
        self.n, self.p = input_shape  # input_tensor shape
        self.m = hidden_unit

        self.w = np.random.normal(0, 1, (self.p, self.m))
        self.b = np.random.normal(0, 1, (self.m, 1))  # TODO change to (1, m)

        self.activation = activation
        self.learnable = learnable

    def train_forward(self, X):
        # Forward Propagation
        self.X = X
        output = np.matmul(self.X, self.w) + self.b.T
        output = self.activation.train_forward(output)

        return output

    def pred_forward(self, X):
        return self.train_forward(X)

    def update_gradient(self, grad, method, minibatch=-1):
        '''
        :param grad: 链式法则传过来的上一层的gradient
        :return:
        '''

        # print("Input gradient", grad.shape)

        self.input_gradient_after_activation = np.multiply(grad, self.activation.gradient['x'])
        self.grad_w = np.matmul(self.X.T, self.input_gradient_after_activation)
        self.grad_x = np.matmul(self.input_gradient_after_activation, self.w.T)
        self.grad_b = self.average_gradient(self.input_gradient_after_activation,
                                            method, minibatch)  # 只有b是需要average gradient

        assert self.grad_w.shape == self.w.shape
        # print(self.grad_b.shape)
        # print(self.b.shape)
        assert self.grad_b.shape == self.b.shape
        assert self.grad_x.shape == self.X.shape

        self.__model_gradient = self.grad_x

    @property
    def params(self):
        return {'w': self.w,
                'b': self.b}

    @property
    def gradient(self):
        '''

        :return: X -- model gradient for backprop
        '''
        return {"w": self.grad_w,
                "b": self.grad_b,
                "x": self.grad_x,
                "back": self.grad_x}


# class Input(Layer):
#     def __init__(self, input_shape):
#         super().__init__()
#         self.n, self.p = input_shape  # input_tensor shape
#
#     def train_forward(self, input_tensor):
#         self.input_tensor = input_tensor
#         return input_tensor
