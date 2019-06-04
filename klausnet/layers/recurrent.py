from klausnet.layers import base
import numpy as np


# RNN
class SimpleRNNCell():

    """
    这个layer只是单纯的一个RNN Cell，
    由于RNN 这块需要两个向前传播的gradient
    同时还需要考虑到loss func不同的不同赢来的不同的RNN Cell 内部的变化
    """

    def __init__(self, input_shape, hidden_units, output_units, activation):

        '''
        :param hidden_units:    # of hidden units
        :param activation:      # of activations
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

    def forward(self, X, h):
        self.X = X
        self.a = self.b.T + h * self.w + X * self.u
        self.h = np.tanh(self.a)
        self.o = self.c.T + self.h * self.v
        sum_e_o = np.sum(np.exp(self.o))
        y = np.exp(self.o) / sum_e_o
        return y, self.h

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

        # da: n x m
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
        # self.dx = np.multiply(self.da, self.u)  # nxm mxp
        # x 的梯度没有任何用处

    @property
    def gradient(self):

        '''
        注意，RNN 在前向传播的时候，传播的是dh，hidden state，
        我们需要与之前的gradient全部对应，
        所以把所有的 gradient['x'] 新建一个变为 gradient['back'].
        :return:
        '''

        return {'w': self.dw,
                'b': self.db,
                'c': self.dc,
                'v': self.dv,
                'u': self.du,
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

class RNN(base.Layer):
    def __init__(self, input_shape, number_of_RNN_cell ,hidden_units,
                 output_units, activation, return_sequence, return_state):

        '''
        :param input_shape:      d x 1
        :param RNN_cell:         Integer. Number of RNN cells
        :param hidden_units:     Integer. Number of hidden cells 向右的输出
        :param output_units:     Integer. Output Units, 向上的输出
        :param activation:       Class Activation
        :param return_sequences: Boolean. Whether to return the last output in the output sequence,
                                          or the full sequence
        :param return_state:     Boolean. Whether to return the last state in addition to the output
        '''

        super().__init__()
        self.input_shape = input_shape
        self.p = number_of_RNN_cell
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.return_sequence = return_sequence
        self.return_state = return_state
        self.activation = activation
        self.RNN_cell = SimpleRNNCell(input_shape=self.input_shape,
                                      hidden_units=self.hidden_units,
                                      output_units=self.output_units,
                                      activation=self.activation)

    def train_forward(self, X):

        '''
        :param X: n x p ; p 应该与 number_of_RNN_cell 保持一致
        :return:
        '''

        self.X = X
        self.n = X.shape[0]
        assert self.p == X.shape[1]

        # Initialize a hidden param
        self.h = np.random.normal(0, 1, (self.n, self.hidden_units))
        self.y = np.zeros((self.p, 1))
        X = self.X

        for _ in range(self.p):
            y, self.h = self.RNN_cell.forward(X, self.h)
