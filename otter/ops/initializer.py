from otter.dam.structure import Variable
import numpy as np


def ones(shape, dtype, lchild=None, rchild=None):
    return Variable(np.ones(shape, dtype), lchild=lchild, rchild=rchild)


def zeros(shape, dtype, lchild=None, rchild=None):
    return Variable(np.zeros(shape, dtype), lchild=lchild, rchild=rchild)


def normal(mu, sigma, shape, lchild=None, rchild=None):
    return Variable(np.random.normal(mu, sigma, shape), lchild=lchild, rchild=rchild)

