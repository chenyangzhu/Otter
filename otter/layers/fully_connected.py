from otter.layers import base
import numpy as np
from ..dam.structure import Variable


class Dense(base.Layer):
    def __init__(self, output_shape, input_shape, activation, learnable=True):

        '''
        :param output_shape:   # of hidden units
        :param input_shape:   input shape, excluding n
        :param activation:    an activation function
        :param learnable:     learnable=True，则在gradient descent 的时候更新这层的param，如果False，则不更新
        '''

        super().__init__()
        # Normal initialization
        self.p = input_shape  # input_tensor shape
        self.m = output_shape
        self.w = Variable(np.random.normal(0, 1, (self.p, self.m)), trainable=True)
        self.b = Variable(np.random.normal(0, 1, (self.m, 1)),
                          trainable=True, param_share=True)

        self.activation = activation
        self.learnable = learnable

    def train_forward(self, x: Variable):
        # Forward Propagation
        self.x = x
        output = x.dot(self.w).add(self.b.T())
        output = self.activation(output)
        return output

    def pred_forward(self, X):
        return self.train_forward(X)
