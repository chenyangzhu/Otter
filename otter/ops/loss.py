import numpy as np
from otter import Variable


def mean_squared_error(y: Variable, yhat: Variable):
    assert y.shape == yhat.shape

    output = ((y - yhat) ** 2).average()

    return output


def sparse_categorical_crossentropy(y: Variable, yhat: Variable):
    """
    :param y:       n x p softmaxed matrix
    :param yhat:    n x 1 integer encoded
    :return:
    """

    sliced = yhat.slice(y.value.reshape((len(y.value),)), axis=1)
    maxi = sliced.log().average().neg()

    return maxi


def sparse_categorical_accuracy(y: Variable, yhat: Variable):
    """
    Notice that this function is not differentiable
    so the return is not a Variable, but only a float number
    """
    argmax_y = np.argmax(yhat.value, axis=1)
    long_y = y.reshape((y.shape[0],)).value

    return np.average(long_y == argmax_y)
