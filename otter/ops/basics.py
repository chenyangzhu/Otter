
from otter import ops as ops
from otter.dam.structure import Variable
from .._hyperparam import *
import numpy as np

"""
Self Calculations
"""
def inv(x: Variable):
    output = Variable(1 / x.value, lchild=x)
    output.back_prop = back_inv
    return output


def back_inv(x: Variable):
    x.lchild.update_gradient(neg(multiply(x.gradient, x ** Variable(np.array(2)))))


def safe_inv(x: Variable):
    return inv(x + Variable(np.array(EPSILON)))


def neg(x: Variable):
    return Variable(np.array(0)) - x


def T(x: Variable):  # Checked
    output = Variable(x=x.value.T, lchild=x)

    output.back_prop = back_T
    return output


def back_T(x: Variable):
    x.lchild.update_gradient(x.gradient.T)


"""
Two variable additions
"""


def __add__(x: Variable, y: Variable):
    output = Variable(x.value + y.value, lchild=x, rchild=y)
    output.back_prop = back_add
    return output


def add(x: Variable, y: Variable):
    output = Variable(x.value + y.value, lchild=x, rchild=y)
    output.back_prop = back_add
    return output


def back_add(x: Variable):
    x.lchild.update_gradient(x.gradient)
    x.rchild.update_gradient(x.gradient)


def __sub__(x: Variable, y: Variable):
    output = Variable(x.value - y.value, lchild=x, rchild=y)
    output.back_prop = back_sub
    return output


def sub(x: Variable, y: Variable):
    output = Variable(x.value - y.value, lchild=x, rchild=y)
    output.back_prop = back_sub
    return output


def back_sub(x: Variable):
    x.lchild.update_gradient(x.gradient)
    x.rchild.update_gradient(ops.neg(x.gradient))


def dot(x: Variable, y: Variable):
    output = Variable(np.dot(x.value, y.value), lchild=x, rchild=y)
    output.back_prop = back_dot
    return output


def back_dot(x: Variable):
    x.lchild.update_gradient(dot(x.gradient, x.rchild.T))
    x.rchild.update_gradient(dot(x.lchild.T, x.gradient))


def multiply(x: Variable, y: Variable):
    assert isinstance(x, Variable)
    assert isinstance(y, Variable)
    assert x.shape == y.shape

    output = Variable(np.multiply(x.value, y.value), lchild=x, rchild=y)
    output.back_prop = back_multiply
    return output


def back_multiply(x: Variable):
    x.lchild.update_gradient(ops.multiply(x.gradient, x.rchild))
    x.rchild.update_gradient(ops.multiply(x.gradient, x.lchild))


def __pow__(x: Variable, power: Variable):
    output = Variable(x.value ** power, lchild=x, rchild=power)
    output.back_prop = back_pow
    return output


def pow(x: Variable, power: Variable):
    output = Variable(x.value ** power.value, lchild=x, rchild=power)
    output.back_prop = back_pow
    return output


def back_pow(x: Variable):
    # TODO add right child gradient
    x.lchild.update_gradient(ops.dot(x.rchild, ops.multiply(x.lchild ** (x.rchild - Variable(np.ones(1))), x.gradient)))


"""
More advanced self calculation
"""


def maximum(x: Variable, axis=None):
    print(x.shape)

    mask = Variable(np.zeros(x.shape))

    if axis is not None:
        max_idx = np.argmax(x.value, axis=axis)

        # TODO Caution: Now we'll only do 2D cases
        if axis == 1:
            mask.value[np.arange(mask.shape[0]), max_idx] = 1
        elif axis == 0:
            mask.value[max_idx, np.arange(mask.shape[1])] = 1
    else:
        print(x.value)
        max_idx = np.argmax(x.value)
        print(max_idx)
        mask.value[max_idx] = 1

    output = multiply(x, mask)

    return output


def sum(x: Variable, axis=None):
    """
    :param axis: if axis = None, return global sum
    :return:
    """
    if axis is not None:
        '''
        When axis is not None, by default the sum function in numpy will return
        a shape probably (1,) for 2-dim matrix. This is true, because all vectors should be a column vector.

        We'll be good in forward propagation, so we will leave the shape unchanged.
        However, this will be very dangerous for later back-prop, so we need to reshape
        the coming gradients to the correct form. This could be seen in later back-prop functions.
        '''
        # Finding the correct shape
        # Notice that whenever our axis is, the output shape should have one on the corresponding axis,
        # and the original shape on other axis's.
        output_shape = list(x.shape)
        output_shape[axis] = 1
        output = Variable(np.sum(x.value, axis=axis).reshape(output_shape),
                          lchild=x)

    else:
        '''
        When axis is None, the value returns to a scalar. While we keep it as a Variable type,
        it should be a better case to have it as a () dimension
        '''
        output = Variable(np.sum(x.value).reshape(()), lchild=x)

    output.sum_grad_parser = {"axis": axis, "shape": x.shape}
    output.back_prop = back_sum
    return output


def back_sum(x: Variable):
    axis = x.sum_grad_parser['axis']
    shape = x.sum_grad_parser['shape']
    # TODO Notice that this is just a 2D case
    if axis is not None:
        # We need to reshape gradients
        gradient_shape = list(shape)
        x.lchild.update_gradient(ops.repeat(x.gradient, gradient_shape[axis], axis=axis))
    else:
        x.lchild.update_gradient(dot(Variable(np.ones(shape)), x.gradient))


def average(x: Variable, axis=None):
    if axis is not None:
        output_shape = list(x.shape)
        output_shape[axis] = 1

        output = Variable(np.average(x.value, axis=axis).reshape(output_shape), lchild=x)
    else:
        output = Variable(np.average(x.value), lchild=x)

    output.average_grad_parser = {"axis": axis,
                                       "shape": x.shape}
    output.back_prop = back_average
    return output


def back_average(x: Variable):
    axis = x.average_grad_parser['axis']
    shape = x.average_grad_parser['shape']

    if axis is None:
        x.lchild.update_gradient(ops.dot(ops.dot(Variable(np.ones(shape)), x.gradient), Variable(1 / np.prod(shape))))
    else:
        gradient_shape = list(shape)
        gradient_shape[axis] = 1
        x.gradient = ops.reshape(x.gradient, gradient_shape)
        x.lchild.update_gradient(np.ones(shape) * x.gradient / shape[axis])