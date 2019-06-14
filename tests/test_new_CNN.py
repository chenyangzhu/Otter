import numpy as np
import matplotlib.pyplot as plt

from otter.dam.structure import Variable
from otter.dam.graph import Graph
from otter.layers.convolution import MaxPooling2D, Conv2D, Flatten
from otter.activation import softmax, sigmoid, relu
from otter.layers.common import Dense, BatchNormalization
from otter.optimizer import GradientDescent
from otter.loss import sparse_categorical_crossentropy

np.random.seed(2020)

with Graph() as g:
    n = 10
    c = 2
    x_len = 8
    y_len = 8
    m = 10

    x = Variable(np.random.normal(0, 1, (n, c, x_len, y_len)))
    y = Variable(np.random.randint(0, m-1, (n, 1)))

    layer1 = Conv2D(input_shape=(c, x_len, y_len),
                    filters=3,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding=None,
                    activation=relu,
                    trainable=True)

    # layer2 = MaxPooling2D(input_shape=layer1.output_shape,
    #                       pool_size=(5, 5),
    #                       strides=(1, 1),
    #                       padding=None)

    # layer2 = BatchNormalization()

    layer3 = Conv2D(input_shape=layer1.output_shape,
                    filters=3,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding=None,
                    activation=relu,
                    trainable=True)

    layer4 = Flatten()

    layer5 = Dense(output_shape=10,
                   activation=softmax,
                   trainable=True)
    optimizer = GradientDescent(0.5)

    loss_list = []

    for _ in range(100):
        layer1_output = layer1.train_forward(x)
        # layer2_output = layer2.train_forward(layer1_output)
        layer3_output = layer3.train_forward(layer1_output)

        layer4_output = layer4.train_forward(layer3_output)
        layer5_output = layer5.train_forward(layer4_output)

        loss = sparse_categorical_crossentropy(y, layer5_output)

        g.update_gradient_with_optimizer(loss, optimizer)
        print(layer1_output.gradient)
        loss_list.append(loss.value)

plt.plot(loss_list)
plt.show()