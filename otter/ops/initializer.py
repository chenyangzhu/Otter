from otter.dam.structure import Variable
import numpy as np


def ones(shape, dtype):
    return Variable(np.ones(shape, dtype))


def zeros(shape, dtype, *args, **kwargs):
    return Variable(np.zeros(shape, dtype), dtype, args, kwargs)

