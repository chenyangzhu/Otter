import klausnet.layers
import numpy as np


class Dense(klausnet.layers.Layer):
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
        self.b = np.random.normal(0, 1, (self.n, self.m))

        self.activation = activation
        self.learnable = learnable

    def forward(self, X):
        # Forward Propagation
        self.X = X
        output = np.matmul(self.X, self.w) + self.b
        output = self.activation.forward(output)

        return output

    def update_gradient(self, grad):
        '''
        :param grad: 链式法则传过来的上一层的gradient
        :return:
        '''

        # print("Input gradient", grad.shape)
        self.input_gradient_after_activation = np.multiply(grad, self.activation.gradient)
        self.grad_w = np.matmul(self.X.T, self.input_gradient_after_activation)
        self.grad_X = np.matmul(self.input_gradient_after_activation, self.w.T)
        self.grad_b = self.input_gradient_after_activation

        assert self.grad_w.shape == self.w.shape
        assert self.grad_b.shape == self.b.shape
        assert self.grad_X.shape == self.X.shape

        self.__model_gradient = self.grad_X

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
                "X": self.grad_X}


# class Input(Layer):
#     def __init__(self, input_shape):
#         super().__init__()
#         self.n, self.p = input_shape  # input_tensor shape
#
#     def forward(self, input_tensor):
#         self.input_tensor = input_tensor
#         return input_tensor

