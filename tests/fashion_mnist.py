import numpy as np
import gzip
import matplotlib.pyplot as plt

from otter.dam.structure import Variable
from otter.dam.graph import Graph
from otter.layers.convolution import Conv2D, Flatten
from otter.ops.activation import softmax, sigmoid, relu
from otter.layers.common import Dense
from otter.optimizer import GradientDescent
from otter.ops.loss import sparse_categorical_crossentropy
from otter.utils import *

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

    np.random.seed(2020)

    (x_train, y_train), (x_test, y_test) = read_data()

    x_train = x_train[:100]
    y_train = y_train[:100]

    avg = np.average(x_train)
    sqrt = np.sqrt(np.var(x_train))
    x_train = (x_train - avg) / sqrt

    n, x_dim, y_dim = x_train.shape  # 60000, 28, 28
    c = 1
    m = 1
    x_train = x_train.reshape(n, c, x_dim, y_dim)
    y_train = y_train.reshape(n, m)

    x_train = Variable(x_train)
    y_train = Variable(y_train)

    layer1 = Conv2D(in_channel=c,
                    out_channel=12,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    activation=sigmoid,
                    bias=False)

    layer2 = Conv2D(in_channel=12,
                    out_channel=8,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    activation=sigmoid,
                    bias=False)

    layer3 = Flatten()

    layer4 = Dense(output_shape=64,
                   activation=sigmoid)

    # layer5 = Dense(output_shape=32,
    #                activation=sigmoid)

    layer6 = Dense(output_shape=10,
                   activation=softmax)

    optimizer = GradientDescent(0.5)

    loss_list = []
    g = Graph()

    for i in range(4000):
        print(i)
        a = layer1.forward(x_train)
        b = layer2.forward(a)
        c = layer3.forward(b)
        d = layer4.forward(c)
        f = layer6.forward(d)
        loss = sparse_categorical_crossentropy(y_train, f)

        g.update_gradient_with_optimizer(loss, optimizer)

        loss_list.append(loss.value)

        if i % 5 == 0:
            plt.clf()
            plt.plot(loss_list)
            plt.show()
