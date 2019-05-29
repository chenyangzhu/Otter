"""
Layers
- Dense
- CNN
- RNN
"""


from klausnet.loss import *

class Layer():
    def __init__(self):
        pass

    def forward(self, input):
        pass

    @property
    def params(self):
        return 0

    def update_gradient(self, grad):
        pass


class Dense(Layer):
    def __init__(self, hidden_unit, input_shape, activation):

        '''
        :param hidden_unit:   # of hidden units
        :param input_shape:   input shape
        :param activation:    Activation Class
        '''

        super().__init__()
        # Normal initialization
        self.n, self.p = input_shape  # input_tensor shape
        self.m = hidden_unit

        self.w = np.random.normal(0, 1, (self.p, self.m))
        self.b = np.random.normal(0, 1, (self.n, self.m))

        self.activation = activation

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
    def self_gradient(self):
        return {"w": self.grad_w,
                "b": self.grad_b}

    @property
    def model_gradient(self):
        return self.__model_gradient


# class Input(Layer):
#     def __init__(self, input_shape):
#         super().__init__()
#         self.n, self.p = input_shape  # input_tensor shape
#
#     def forward(self, input_tensor):
#         self.input_tensor = input_tensor
#         return input_tensor
