from otter.dam.structure import Variable
import numpy as np
from otter import ops as ops


def clip(x, floor, ceiling):
    """Value clipping"""
    output = Variable(np.clip(x.value, floor, ceiling), lchild=x)
    output.back_prop = back_clip
    return output


def back_clip(x: Variable):
    # Do not change the gradient, directly pass on.
    x.lchild.update_gradient(x.gradient)


def reshape(x: Variable, new_shape):
    output = Variable(x.value.reshape(new_shape), lchild=x)
    output.back_prop = back_reshape
    return output


def back_reshape(x: Variable):
    x.lchild.update_gradient(ops.reshape(x.gradient, x.lchild.shape))


def slice(x: Variable, index, axis):
    """
    This function selects the index from the matrix
    :param index: !! must be a 1d array!!!!! cannot be a matrix!!!!!
    :param axis:
    :return:
    """

    # TODO This is a 2D example
    mask = ops.zeros(x.shape, dtype=np.float64)

    if axis == 0:
        x_index = (index.value, np.arange(x.shape[1]))
        shape = (1, x.shape[1])
    else:
        x_index = (np.arange(x.shape[0]), index.value)
        shape = (x.shape[0], 1)

    output = x.value[x_index].reshape(shape)
    mask.value[x_index] = 1

    output = Variable(output, lchild=x)
    output.slice_grad_parser = {"mask": mask,
                                'axis': axis}
    output.back_prop = back_slice
    return output


def back_slice(x: Variable):
    mask = x.slice_grad_parser['mask']
    axis = x.slice_grad_parser['axis']
    x.lchild.update_gradient(ops.multiply(mask, repeat(x.gradient, mask.shape[axis], axis)))


def repeat(x, repeat_number, axis):

    output = Variable(np.repeat(x.value, repeat_number, axis), lchild=x)
    output.back_prop = back_repeat
    output.repeat_axis = axis

    return output


def back_repeat(x):
    x.lchild.update_gradient(reshape(ops.average(x.gradient, axis=x.repeat_axis), x.lchild.gradient.shape))
