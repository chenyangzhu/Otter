from otter.layers import common
import numpy as np
from otter.dam.structure import Variable
from otter.ops.activation import tanh


# RNN
class SimpleRNNCell():

    """
    Only an RNN cell.
    """

    def __init__(self, hidden_units, output_units, activation):

        """
        :param input_shape:
        :param hidden_units:    # of right output units
        :param output_units:    # of top output units
        :param activation:      # of activations
        """

        super().__init__()
        self.hidden_units = hidden_units    # Hidden units
        self.output_units = output_units    # output units
        self.activation = activation        # Activation for hidden state
        self.initialize = True

        # Initialize all params
        self.w = Variable(np.random.normal(0, 1, (self.hidden_units, self.hidden_units)), trainable=True)
        self.b = Variable(np.random.normal(0, 1, (1, self.hidden_units)),
                          trainable=True, param_share=True)
        self.c = Variable(np.random.normal(0, 1, (1, self.output_units)),
                          trainable=True, param_share=True)
        self.v = Variable(np.random.normal(0, 1, (self.hidden_units, self.output_units)), trainable=True)

    def train_forward(self, x: Variable, h=None):
        """

        :param x: shape: (batch_size, sequence_length, vocab_size)
        :param h:
        :return:
        """
        if self.initialize:
            self.u = Variable(np.random.normal(0, 1, (x.shape[1], self.hidden_units)), trainable=True)
            self.initialize = False

        if h is None:
            # In the first RNNcell, we don't have any hidden layers, so we initialize one
            h = Variable(np.random.normal(0, 1, (x.shape[0], self.hidden_units)))

        xu = x.dot(self.u)
        hw = h.dot(self.w)
        self.a = xu + hw + self.b
        self.h = self.activation(self.a)
        self.o = self.h.dot(self.v) + self.c
        return self.o, self.h

    def predict_forward(self, x: Variable, h=None):
        return self.train_forward(x, h)


class StackedRNNCell():
    def __init__(self, cells, ):



class RNN(common.Layer):
    def __init__(self, cell, number_of_rnn_cell, hidden_units,
                 output_units, activation, return_sequence, return_state):
        """
        This class acted as a layer.

        Args:

            cell:                   if in tuple, means stacked RNN Cell
            number_of_rnn_cell
d

        """

        super().__init__()
        self.p = number_of_rnn_cell
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.return_sequence = return_sequence
        self.return_state = return_state
        self.activation = activation
        self.RNN_cell = SimpleRNNCell(hidden_units=self.hidden_units,
                                      output_units=self.output_units,
                                      activation=self.activation)

    def forward(self, x):

        '''
        :param x:
        :return:
        '''

        self.n = x.shape[0]
        assert self.p == x.shape[1]

        output_list = []
        # Initialize a hidden param
        self.h = Variable(np.random.normal(0, 1, (self.n, self.hidden_units)))

        for _ in range(self.p):
            y, self.h = self.RNN_cell.train_forward(x, self.h)
            output_list.append(y)

        return output_list, self.h


if __name__ == "__main__":

    from otter import Variable
    from otter.dam.graph import Graph
    from otter.ops.activation import softmax
    from otter.optimizer import GradientDescent

    with Graph() as g:
        n = 1000
        p = 64      # Sentence Length
        q = 5       # Prediction Choices
        m = 64      # Embedding Length
        x = Variable(np.random.normal(0, 0.1, (n, p)))
        y = Variable(np.random.randint(0, q-1, (n, p)))
        layer2 = RNN(input_shape=p, number_of_rnn_cell=p,
                     hidden_units=8, output_units=q,
                     activation=softmax, return_sequence=True, return_state=True)

        output, hidden = layer2.forward(x)
        print(len(output))
        print(output[0].shape)

        optimizer = GradientDescent(0.5)

        print("Forward Done.")
        for each in output:
            g.update_gradient_with_optimizer(each, optimizer)
        print("Backward Done")


    # def update_gradient(self, grad, method, minibatch=-1):
    #     '''
    #     :param grad:        [n, output_units]
    #     :param method:
    #     :param minibatch:
    #     :return:
    #     '''
    #
    #     # n x q
    #     self.do = (np.exp(self.o) * np.sum(np.exp(self.o) - np.exp(2 * self.o)))/(np.sum(np.exp(self.o))**2)
    #
    #     # 1 x q
    #     self.dc = np.average(self.do, axis=0).reshape((self.output_units, 1))
    #
    #     # dv: m x q; h: n x m
    #     self.dv = np.matmul(self.h.T, self.do)
    #
    #     # dh: n x m
    #     self.dh = np.matmul(self.do, self.v.T)
    #
    #     # da: n x m
    #     self.da = np.multiply((1 - np.tanh(self.a)), self.dh)
    #
    #     # db: m x 1
    #     self.db = np.average(self.da, axis=0).reshape((self.hidden_units, 1))
    #
    #     # dw: m x m
    #     self.dw = np.multiply(self.h.T, self.da)
    #
    #     # dh: n x m
    #     self.dh = np.multiply(self.da, self.w)
    #
    #     # du: p x m
    #     self.du = np.multiply(self.x.T, self.da)
    #
    #     # dx: n x p
    #     # self.dx = np.multiply(self.da, self.u)  # nxm mxp
    #     # x 的梯度没有任何用处
    #
    # @property
    # def gradient(self):
    #
    #     '''
    #     注意，RNN 在前向传播的时候，传播的是dh，hidden state，
    #     我们需要与之前的gradient全部对应，
    #     所以把所有的 gradient['x'] 新建一个变为 gradient['back'].
    #     :return:
    #     '''
    #
    #     return {'w': self.dw,
    #             'b': self.db,
    #             'c': self.dc,
    #             'v': self.dv,
    #             'u': self.du,
    #             'h': self.dh,
    #             'back': self.dh}
    #

