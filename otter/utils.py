# from sklearn.preprocessing import OneHotEncoder
import numpy as np
from otter.dam.structure import Variable

#
# def int2onehot(vocab_size, x):
#     one_hot_encoder = OneHotEncoder(categories=[range(vocab_size)]).fit(x)
#     one_hot_x = one_hot_encoder.transform(x).toarray()
#
#     return one_hot_x


def last_dim_one_hot(x, vocab_size):

    """
    This function changes the last dimension of input x into one-hot encoding
    :param x: takes whatever dimension, but the last dim must be [0 -> max]
    :return:
    """

    # We first detect the largest value
    shape = list(x.shape)
    shape[-1] = vocab_size
    one_hot = np.zeros(shape)
    # We first reshape one_hot
    reshaped_x = x.reshape(np.prod(shape[:-1]), 1)
    reshaped_one_hot = one_hot.reshape(np.prod(shape[:-1]), vocab_size)
    reshaped_one_hot[np.arange(np.prod(shape[:-1])), np.array(reshaped_x).T[0]] = 1
    reshape_back = reshaped_one_hot.reshape(shape)

    return reshape_back

def l2norm(x):
    return np.sum(x ** 2)