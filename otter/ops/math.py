from otter._hyperparam import *
import numpy as np
from otter.dam.structure import Variable
from otter import ops as ops


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
    output.back_prop = back_log
    return output


def safe_log(x: Variable):
    return log((x + ops.constant(EPSILON)))


def back_log(x):
    x.lchild.update_gradient(ops.multiply(ops.safe_inv(x), x.gradient))
