import numpy as np
from otter.dam.structure import Variable
from otter import ops as ops


def mean_squared_error(y: Variable, yhat: Variable):
    assert y.shape == yhat.shape
    output = ops.average(ops.pow(y - yhat, Variable(np.array(2))))
    return output


def sparse_categorical_crossentropy(y: Variable, yhat: Variable):
    """
    :param y:       n x p softmaxed matrix
    :param yhat:    n x 1 integer encoded
    :return:
    """

    sliced = ops.slice(yhat, ops.reshape(y, (y.shape[0], )), axis=1)
    maxi = ops.neg(ops.average(ops.safe_log(sliced)))
    return maxi


def sparse_categorical_crossentropy_with_softmax(y: Variable, yhat: Variable):

    sliced = ops.slice(yhat, ops.reshape(y, (y.shape[0], )), axis=1)
    sum_sliced = ops.average(sliced)
    exp_yhat = ops.average(ops.clip(ops.safe_log(ops.sum(ops.safe_exp(yhat), axis=1)), -1.5, 1.5))
    return exp_yhat - sum_sliced


def sparse_categorical_accuracy(y: Variable, yhat: Variable):

    """
    Notice that this function is not differentiable
    so the return is not a Variable, but only a float number
    """

    argmax_y = np.argmax(yhat.value, axis=1)
    long_y = ops.reshape(y, (y.shape[0],)).value
    return np.average(long_y == argmax_y)
