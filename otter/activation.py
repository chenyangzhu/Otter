"""
In new activation, all we pass on is a Variable, and we return a Variable, which contains the graph built on the
input Variable.

In this way, we can easily build up the graph and allow the back-prop.

In e.g. softmax, the usual back-prop will not work with two grad-routes
We therefore need to creat a new Variable, and write the gradient rule
in the Variable class.

"""

import numpy as np
from otter.dam.structure import Variable


def sigmoid(x: Variable):
    return x.neg().exp().add(Variable(np.ones(1))).inv()


def softmax(x: Variable, axis=0):
    # The reason to creat a new Variable, is to drop the connection and rewrite the gradients
    # in dam.structure.Variable
    exp_sum_inv = x.exp().sum(axis=axis).inv()
    output_value = x.exp().multiply(exp_sum_inv).value
    output = Variable(output_value, lchild=x)
    output.softmax_grad_parser = {"output": output_value}

    output.back_prop = output.back_softmax
    return output

def relu(x: Variable):

    mapping = Variable((x.value > 0).astype(int))

    output = x.multiply(mapping)

    return output



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