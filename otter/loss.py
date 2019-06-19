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
    # print("++++++++++sliced+++++++++++", sliced)
    # print(sliced.log().sum())
    maxi = sliced.safe_log().average().neg()

    return maxi
