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
    return x.neg().safe_exp().add(Variable(np.ones(1))).inv()


def softmax(x: Variable, axis=0):
    # The reason to creat a new Variable, is to drop the connection and rewrite the gradients
    # in dam.structure.Variable
    # print(x.value)
    M = np.max(x.value)
    # print(M)
    small_x = x.value - M
    # print(small_x)
    exp_small_x = np.exp(small_x)

    '''
    The reason we subtract the maximum value from x
    is to avoid overflow problem when doing exp()
    '''

    exp_sum_inv = 1 / (np.sum(exp_small_x, axis=axis))

    output_value = np.multiply(exp_small_x, exp_sum_inv)

    output = Variable(output_value, lchild=x)
    output.back_prop = output.back_softmax
    return output


def relu(x: Variable):

    mapping = Variable((x.value > 0).astype(int))

    output = x.multiply(mapping)

    return output


def tanh(x: Variable):
    # TODO Solve OOM problem
    M = np.max(x.value)
    output_value = (np.exp(x.value - M) - np.exp(-x.value - M))/(np.exp(x.value - M) + np.exp(-x.value - M))

    output = Variable(output_value, lchild=x)
    output.back_prop=output.back_tanh
    output.tanh_grad_parser = {'M': M,
                               'xvalue': x.value}
    return output