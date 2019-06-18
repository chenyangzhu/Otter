import numpy as np
import matplotlib.pyplot as plt

from otter.dam.structure import Variable
from otter.dam.graph import Graph
from otter.layers.convolution import MaxPooling2D, Conv2D, Flatten
from otter.activation import softmax, sigmoid, relu
from otter.layers.common import Dense, BatchNormalization
from otter.optimizer import GradientDescent
from otter.loss import sparse_categorical_crossentropy

np.random.seed(2019)

with Graph() as g:
    n = 1000
    c = 2
    x_len = 8
    y_len = 8
    m = 2

    x = Variable(np.random.normal(0, 1, (n, c, x_len, y_len)))
    y = Variable(np.random.randint(0, m-1, (n, 1)))

    layer1 = Conv2D(input_shape=(c, x_len, y_len),
                    filters=5,
                    kernel_size=(5, 5),
                    strides=(3, 3),
                    padding=None,
                    activation=sigmoid,
                    trainable=True)

    layer3 = Conv2D(input_shape=layer1.output_shape,
                    filters=3,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding=None,
                    activation=relu,
                    trainable=True)

    layer4 = BatchNormalization()

    layer5 = Flatten()

    layer6 = Dense(output_shape=10,
                   activation=relu,
                   trainable=True)

    optimizer = GradientDescent(1)

    loss_list = []

    for i in range(400):
        a = layer1.train_forward(x)
        c = layer3.train_forward(a)

        d = layer4.train_forward(c)
        e = layer5.train_forward(d)
        f = layer6.train_forward(e)
        # print(f)
        g = softmax(f)
        loss = sparse_categorical_crossentropy(y, g)

        g.update_gradient_with_optimizer(loss, optimizer)
        # print(a.gradient)
        loss_list.append(loss.value)

        if i % 3 == 0:
            plt.plot(loss_list)
            plt.show()