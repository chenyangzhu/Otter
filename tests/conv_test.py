import numpy as np
import gzip
import matplotlib.pyplot as plt

from otter.dam.structure import Variable
from otter.dam.graph import Graph
from otter.layers.convolution import MaxPooling2D, Conv2D, Flatten
from otter.activation import softmax, sigmoid, relu, tanh
from otter.layers.common import Dense, BatchNormalization
from otter.optimizer import GradientDescent
from otter.loss import sparse_categorical_crossentropy


def read_data():
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    paths = []
    for i in range(len(files)):
        paths.append('../dataset/fashion-mnist/' + files[i])

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = read_data()

    x_train = x_train[0]
    y_train = y_train[0]

    avg = np.average(x_train)
    sqrt = np.sqrt(np.var(x_train))
    x_train = (x_train - avg) / sqrt

    x_dim, y_dim = x_train.shape  # 60000, 28, 28
    # print(x_dim)
    c = 1
    m = 1
    x_train = x_train.reshape(1, c, x_dim, y_dim).astype(np.float64)
    y_train = y_train.reshape(1, m).astype(np.float64)

    x_train = Variable(x_train)
    y_train = Variable(y_train)

    with Graph() as g:
        layer1 = Conv2D(input_shape=(c, x_dim, y_dim),
                        filters=16,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding=None,
                        activation=relu,
                        trainable=True)

        a = layer1.train_forward(x_train)
        # print(a.shape)
        print(a)
        # plt.imshow(a.value)
        # plt.show()
        g.update_gradient(a)