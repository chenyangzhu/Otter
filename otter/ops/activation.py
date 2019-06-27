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
    return x.neg().safe_exp().add(Variable(np.ones(1))).safe_inv()


def softmax(x: Variable, axis=1):
    M = Variable(x.maximum().value)  # M must only be a constant
    small_x = x - M
    exp_small_x = small_x.safe_exp()
    inv_sum_exp_small_x = exp_small_x.sum(axis=axis).safe_inv()
    long_inv_sum_exp_small_x = inv_sum_exp_small_x.repeat(x.shape[1], axis=axis)
    output = exp_small_x.multiply(long_inv_sum_exp_small_x)

    '''
    The reason we subtract the maximum value from x
    is to avoid overflow problem when doing exp()
    '''

    # exp_sum_inv = 1 / (np.sum(exp_small_x.value, axis=axis))
    #
    # output_value = np.multiply(exp_small_x.value, exp_sum_inv)
    #
    # output = Variable(output_value, lchild=x)
    # output.back_prop = output.back_softmax
    return output


def relu(x: Variable):

    mapping = Variable((x.value > 0).astype(int))
    output = x.multiply(mapping)

    return output


def tanh(x: Variable):
    M = np.average(x.value)
    print(np.max(x.value - M))
    print(np.min(x.value - M))
    output_value = (np.exp(x.value - M) - np.exp(-x.value - M))/(np.exp(x.value - M) + np.exp(-x.value - M))
    output = Variable(output_value, lchild=x)

    output.back_prop = output.back_tanh
    output.tanh_grad_parser = {'M': M,
                               'xvalue': x.value}
    return output
