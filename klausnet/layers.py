"""
Layers
- Dense
- CNN
- RNN

### 目前，Dense Layer是自带一个activation的！！！！！
但我们也可以把activation 作为一个新的layer，直接放到model里。
"""

from klausnet.loss import *

class Layer():
    def __init__(self):
        pass

    def forward(self, X):
        '''
        如果可以的话，gradient尽量从forward中update
        但遇到activation内置的情况，可以在update_gradient中嵌入。
        :param X:
        :return:
        '''
        pass

    def update_gradient(self, grad):
        '''
        用来跟新gradient，（用于back prop的时候）
        :param grad:
        :return:
        '''
        pass

    @property
    def params(self):
        return 0

    @property
    def gradient(self):
        return 0

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


class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides,
                 padding, activation):
        '''
        :param filters:         kernel 的维度
        :param kernel_size:     kernel 的大小
        :param strides:         //
        :param padding:         //
        :param activation:      //
        '''
        super().__init__()

        # Initialize the kernel
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel = np.random.normal(0, 1, self.kernel_size)

        self.strides = strides
        self.padding = padding
        self.activation = activation

    def forward(self, X):
