from otter import ops as ops
from otter.dam.structure import Variable
from .._hyperparam import *
import numpy as np
import otter as ot


def conv2d(input, filter, strides, padding, data_format='NHWC'):
    pass


def sparse_dot_with_mapping(x, w, mapping, sparse_matrix_height, sparse_matrix_width):
    """
    This function calculates a sparse matrix multiplication of xw

    Args:

        x is a dense matrix, stored as a type Variable
        w is a sparse matrix, but stored in a dense way, as a type Variable
        mapping is the mapping from w to the *fake* sparse matrix which is a list
                containing the index for [(w), (fake sparse matrix)]
    """

    output = ot.zeros(shape=(x.shape[0], sparse_matrix_width), dtype=np.float64)
    output.rchild = w
    output.lchild = x
    output.back_prop = back_sparse_dot_with_mapping

    for each_mapping in mapping:
        index_in_w = each_mapping[0]
        i, j = each_mapping[1]
        output.value[:, j] += x.value[:, i] * w.value[index_in_w]

    output.sparse_dot_with_mapping_grad_parser = {'mapping': mapping}

    return output


def back_sparse_dot_with_mapping(x: Variable):
    """
    This function is the back_propagation of the sparse matrix multiplication
    defined above

    Caution: Notice that this gradient is not updated with the default function.
    """

    mapping = x.sparse_dot_with_mapping_grad_parser['mapping']

    rchild_gradient = ot.zeros(x.rchild.shape, dtype=np.float64)
    lchild_gradient = ot.zeros(x.lchild.shape, dtype=np.float64)

    for each_mapping in mapping:
        index_in_w = each_mapping[0]
        i, j = each_mapping[1]
        rchild_gradient.value[index_in_w] += np.sum(x.lchild.value[:, i] * x.gradient.value[:, j])
        lchild_gradient.value[:, i] += x.rchild.value[index_in_w] * x.gradient.value[:, j]

    x.rchild.update_gradient(rchild_gradient)
    x.lchild.update_gradient(lchild_gradient)
