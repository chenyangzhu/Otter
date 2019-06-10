"""
In new activation, all we pass on is a Variable, and we return a Variable, which contains the graph built on the
input Variable.

In this way, we can easily build up the graph and allow the back-prop.

"""

import numpy as np
from otter.dam.structure import Variable


def sigmoid(x: Variable):
    return x.neg().exp().add(Variable(np.ones(1))).inv()

# class Activation:
#
#     def __init__(self):
#
#         pass
#
#     def train_forward(self, x):
#
#         return x
#
#     def predict_forward(self, x):
#         """
#         By default, we'll return train_forward.
#         The reason to keep track of two lines is that for layers like dropouts, it would be much easier to
#         differentiate from the two paths. The activation layers are also in line with the real layers.
#
#         We'll omit this function in later child classes, as it is declared universally in the parent class,
#         unless predict and train are different in some special cases.
#
#         :param x: Variable
#         :return:  Variable
#         """
#
#         return self.train_forward(x)

# class Softmax(Activation):
#     def __init__(self):
#         super().__init__()
#
#     def train_forward(self, x):
#         # TODO Softmax
#         pass
#
# class Tanh(Activation):
#     def __init__(self):
#         super().__init__()
#
#     def train_forward(self, x):
#         # TODO tanh
#         pass
#
# class Linear(Activation):
#     def __init__(self):
#         super().__init__()
#
#     def train_forward(self, x):
#         return x
#
#
# class Relu(Activation):
#     def __init__(self):
#         super().__init__()
#
#     def train_forward(self, x):
#         # TODO Relu
#         pass