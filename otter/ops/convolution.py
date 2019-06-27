import numpy as np
from ..dam.structure import Variable
import otter


def conv2d(input, filter, strides, padding, data_format='NHWC'):

    pass


def sparse_dot_with_mapping(x, w, mapping,
                            sparse_matrix_height, sparse_matrix_width):
    """
    This function calculates a sparse matrix multiplication of xw

    Args:

        x is a dense matrix, stored as a type Variable
        w is a sparse matrix, but stored in a dense way, as a type Variable
        mapping is the mapping from w to the *fake* sparse matrix which is a list
                containing the index for [(w), (fake sparse matrix)]
    """

    output = otter.zeros(shape=(x.shape[0], sparse_matrix_width), dtype=np.float32)
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

    x.rchild.gradient = np.zeros_like(x.rchild.gradient)
    x.lchild.gradient = np.zeros_like(x.lchild.gradient)

    for each_mapping in mapping:
        index_in_w = each_mapping[0]
        i, j = each_mapping[1]
        x.rchild.gradient[index_in_w] += np.sum(x.lchild.value[:, i] * x.gradient.value[:, j])
        x.lchild.gradient[:, i] += x.rchild.value[index_in_w] * x.gradient.value[:, j]

    x.rchild.gradient = Variable(x.rchild.gradient)
    x.lchild.gradient = Variable(x.lchild.gradient)

