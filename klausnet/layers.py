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

    def update_model_gradient(self, grad):
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

    def forward(self, input_tensor):
        # Forward Propagation
        self.input_tensor = input_tensor
        output = np.matmul(self.input_tensor, self.w) + self.b
        output = self.activation.forward(output)

        self.calculate_self_gradient()

        return output

    def calculate_self_gradient(self):
        # Calculate Self Gradients
        self.grad_w = np.matmul(self.input_tensor.T, self.activation.gradient)
        assert self.grad_w.shape == self.w.shape
        self.grad_b = self.activation.gradient
        assert self.grad_b.shape == self.b.shape

        self.grad_x = np.matmul(self.activation.gradient, self.w.T)
        assert self.grad_x.shape == self.input_tensor.shape

        # Model gradients ready for back-prop
        self.model_gradient = self.grad_x
        # model gradient 就是我要往后传播的东西

    def update_model_gradient(self, grad):
        '''
        :param grad: 链式法则传过来的上一层的gradient
        :return:
        '''
        self.model_gradient = np.matmul(self.model_gradient.T, grad)

    @property
    def params(self):
        return {'w': self.w,
                'b': self.b}

    @property
    def self_gradient(self):
        return {"w": self.grad_w,
                "b": self.grad_b}


class Input(Layer):
    def __init__(self, input_shape):
        super().__init__()
        self.n, self.p = input_shape  # input_tensor shape

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return input_tensor
