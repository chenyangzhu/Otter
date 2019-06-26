import numpy as np
import gzip
import matplotlib.pyplot as plt

from otter.dam.structure import Variable
from otter.dam.graph import Graph
from otter.layers.convolution import Conv2D, Flatten, MaxPooling2D
from otter.ops.activation import softmax, sigmoid, relu
from otter.layers.common import Dense
from otter.optimizer import *
from otter.ops.loss import *
from otter.utils import *
from otter.model import Model

from tqdm import tqdm
import tensorflow as tf


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

    np.random.seed(2019)

    (x_train, y_train), (x_test, y_test) = read_data()

    x_train = x_train[:20000]
    y_train = y_train[:20000]

    avg = np.average(x_train)
    sqrt = np.sqrt(np.var(x_train))
    x_train = (x_train - avg) / sqrt

    n, x_dim, y_dim = x_train.shape  # 60000, 28, 28
    c = 1
    m = 1
    x_train = x_train.reshape(n, c, x_dim, y_dim)
    y_train = y_train.reshape(n, m)

    conv1 = Conv2D(out_channel=8,
                   kernel_size=(5, 5),
                   stride=(2, 2),
                   activation=relu)

    # pooling1 = MaxPooling2D(kernel_size=(3, 3),
    #                         stride=(2, 2))

    # conv2 = Conv2D(out_channel=4,
    #                kernel_size=(3, 3),
    #                stride=(2, 2),
    #                activation=relu)

    flatten = Flatten()
    dense2 = Dense(output_shape=10,
                   activation=softmax)

    optimizer = GradientDescent(0.8)

    loss_list = []
    acc_list = []
    norm1_list = []
    norm2_list = []
    g = Graph()

    iteration = 1000
    batch_size = 256
    total_epoch = int(n / batch_size)

    for it_idx in range(iteration):
        print(f"The {it_idx}th iteration.")
        for epoch in tqdm(range(total_epoch)):

            x = x_train[epoch*batch_size: (epoch+1) * batch_size]
            y = y_train[epoch*batch_size: (epoch+1) * batch_size]

            x = Variable(x)
            y = Variable(y)

            a = conv1.forward(x)
            c = flatten.forward(a)
            f = dense2.forward(c)

            loss = sparse_categorical_crossentropy(y, f)
            acc = sparse_categorical_accuracy(y, f)

            g.update_gradient_with_optimizer(loss, optimizer)
            loss_list.append(loss.value)
            acc_list.append(acc)
            norm1_list.append(l2norm(conv1.w.value))
            norm2_list.append(l2norm(dense2.w.value))

            print(f" acc:{acc}, loss:{loss.value}")
            print(conv1.w.gradient)
            # print(dense2.w.gradient)
            # print(f)

            if epoch % 5 == 0:
                plt.clf()
                plt.plot(loss_list)
                plt.show()

                #
                # plt.plot(acc_list)
                # plt.show()
                #
                # plt.plot(norm1_list)
                # plt.show()
                #
                # plt.plot(norm2_list)
                # plt.show()
