
from .basics import *


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
    x.lchild.update_gradient(x.gradient.reshape(x.lchild.shape))


def slice(x: Variable, index, axis):
    """
    This function selects the index from the matrix
    :param index: !! must be a 1d array!!!!! cannot be a matrix!!!!!
    :param axis:
    :return:
    """

    # TODO This is a 2D example
    mask = np.zeros_like(x.value)
    if axis == 0:
        output = x.value[index,
                            np.arange(x.shape[1])].reshape((1, x.shape[1]))
        mask[index, np.arange(x.shape[1])] = 1
    elif axis == 1:
        output = x.value[np.arange(x.shape[0]), index].reshape((x.shape[0], 1))
        mask[np.arange(x.shape[0]), index] = 1
        # print(index)

    output = Variable(output, lchild=x)
    output.slice_grad_parser = {"mask": mask}
    output.back_prop = back_slice
    return output


def back_slice(x: Variable):
    x.lchild.update_gradient(multiply(x.slice_grad_parser['mask'], x.gradient))


def repeat(x, repeat_number, axis):

    output = Variable(np.repeat(x.value, repeat_number, axis), lchild=x)
    output.back_prop = back_repeat
    output.repeat_axis = axis

    return output


def back_repeat(x):
    x.lchild.update_gradient(reshape(average(x.gradient, axis=x.repeat_axis), x.lchild.gradient.shape))
