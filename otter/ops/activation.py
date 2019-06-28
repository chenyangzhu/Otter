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
from otter import ops as ops


def sigmoid(x: Variable):
    return ops.safe_inv(ops.safe_exp(ops.neg(x)) + Variable(np.ones(1)))


def softmax(x: Variable, axis=1):
    M = ops.maximum(x).copy()  # M must only be a constant
    small_x = x - M
    exp_small_x = ops.safe_exp(small_x)
    inv_sum_exp_small_x = ops.safe_inv(ops.sum(exp_small_x, axis=axis))
    long_inv_sum_exp_small_x = ops.repeat(inv_sum_exp_small_x, x.shape[1], axis=axis)
    output = ops.multiply(exp_small_x, long_inv_sum_exp_small_x)
    return output


def relu(x: Variable):
    mapping = Variable((x.value > 0).astype(int))
    output = ops.multiply(x, mapping)
    return output


# # TODO write tanh
# def tanh(x: Variable):
#     M = np.average(x.value)
#     output_value = (np.exp(x.value - M) - np.exp(-x.value - M))/(np.exp(x.value - M) + np.exp(-x.value - M))
#     output = Variable(output_value, lchild=x)
#
#     output.back_prop = back_tanh
#     output.tanh_grad_parser = {'M': M,
#                                'xvalue': x.value}
#     return output
# #
# # def back_tanh(self):
# #     self.lchild.gradient = np.multiply(1 - self.lchild.value ** 2, self.gradient)
