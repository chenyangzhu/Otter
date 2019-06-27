import numpy as np
from otter._hyperparam import *
import otter.ops as ops
from otter.dam.structure import Variable


def exp(x: Variable) -> Variable:

    output = Variable(np.exp(x.value), lchild=x)
    output.back_prop = back_exp

    return output


def safe_exp(x: Variable):
    return ops.exp(ops.clip(x, -VALUE_CLIPPING_THRESHOLD, VALUE_CLIPPING_THRESHOLD))


def back_exp(x: Variable):
    x.lchild.update_gradient(ops.multiply(x.gradient, x))


def log(x: Variable) -> Variable:
    output = Variable(np.log(x.value), lchild=x)
    inverse = 1 / x.value

    output.log_grad_parser = {"inverse": inverse}
    output.back_prop = output.back_log
    return output


def safe_log(x: Variable):
    return log((x + Variable(EPSILON)))


def back_log(x):
    x.lchild.update_gradient(ops.multiply(x.log_grad_parser['inverse'],
                                          x.gradient))

