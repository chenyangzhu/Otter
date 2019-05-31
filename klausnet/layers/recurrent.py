from klausnet.layers import base
import numpy as np


# RNN
class SimpleRNNCell(base.Layer):
    """
    这个layer只是单纯的一个RNN Cell，
    """
    def __init__(self, input_shape, hidden_units, output_units, activation):
        '''
        :param hidden_units: # of hidden units
        :param activation: # of activations
        '''

        super().__init__()
        self.n, self.p = input_shape        # Input shape
        self.hidden_units = hidden_units    # Hidden units
        self.output_units = output_units    # output units
        self.activation = activation

        # Initialize all params
        self.u = np.random.normal(0, 1, (self.p, self.hidden_units))
        self.w = np.random.normal(0, 1, (self.hidden_units, self.hidden_units))
        self.b = np.random.normal(0, 1, (self.hidden_units, 1))
        self.c = np.random.normal(0, 1, (self.output_units, 1))
        self.v = np.random.normal(0, 1, (self.hidden_units, self.output_units))

    def forward_prop(self, X, h):
        self.X = X
        self.a = self.b.T + h * self.w + X * self.u
        self.h = np.tanh(self.a)
        self.o = self.c.T + self.h * self.v
        sum_e_o = np.sum(np.exp(self.o))
        y = np.exp(self.o) / sum_e_o
        return y

    def update_gradient(self, grad, method, minibatch=-1):
        '''

        :param grad:        [n, output_units]
        :param method:
        :param minibatch:
        :return:
        '''

        # n x q
        self.do = (np.exp(self.o) * np.sum(np.exp(self.o) - np.exp(2 * self.o)))/(np.sum(np.exp(self.o))**2)

        # 1 x q
        self.dc = np.average(self.do, axis=0).reshape((self.output_units, 1))

        # dv: m x q; h: n x m
        self.dv = np.matmul(self.h.T, self.do)

        # dh: n x m
        self.dh = np.matmul(self.do, self.v.T)

        # da: nxm
        self.da = np.multiply((1 - np.tanh(self.a)), self.dh)

        # db: m x 1
        self.db = np.average(self.da, axis=0).reshape((self.hidden_units, 1))

        # dw: m x m
        self.dw = np.multiply(self.h.T, self.da)

        # dh: n x m
        self.dh = np.multiply(self.da, self.w)

        # du: p x m
        self.du = np.multiply(self.X.T, self.da)

        # dx: n x p
        self.dx = np.multiply(self.da, self.u)  # nxm mxp

    @property
    def gradient(self):

        '''
        注意，RNN 在前向传播的时候，传播的是dh，hidden state，
        我们需要与之前的gradient全部对应，所以把所有的 gradient['x'] 新建一个变为 gradient['back']
        :return:
        '''

        return {'w': self.dw,
                'b': self.db,
                'c': self.dc,
                'v': self.dv,
                'u': self.du,
                'x': self.dx,
                'h': self.dh,
                'back': self.dh}

    @property
    def hidden_state(self):
        return self.h

    @property
    def params(self):
        return {'w': self.w,
                'b': self.b,
                'c': self.c,
                'v': self.v,
                'u': self.u}