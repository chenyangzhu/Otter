import numpy as np
from ..dam.structure import Variable


def sparse_dot_with_mapping(x, w, mapping, sparse_matrix_height, sparse_matrix_width):
    """
    This function calculates a sparse matrix multiplication of xw

    Args:

        x is a dense matrix, stored as a type Variable
        w is a sparse matrix, but stored in a dense way, as a type Variable
        mapping is the mapping from w to the *fake* sparse matrix which is a list
                containing the index for [(w), (fake sparse matrix)]
    """

    output = Variable(np.zeros(x.shape[0], sparse_matrix_width),
                      lchild=x, rchild=w)

    for each_mapping in mapping:

        index_in_w = each_mapping[0]
        i, j = each_mapping[1]

        # the i,j th element in the sparse matrix,
        # is multiplied by the ith column from the x matrix
        # and is added to jth column in the output column.

        output.value[:, j] += x[:, i] * w[index_in_w]

    return output
