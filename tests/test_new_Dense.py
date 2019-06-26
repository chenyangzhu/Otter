import numpy as np
import matplotlib.pyplot as plt

from otter import Variable
from otter.ops.loss import mean_squared_error
from otter.dam.graph import Graph
from otter.ops.activation import sigmoid
from otter.layers.common import Dense
from otter.optimizer import *

np.random.seed(2019)


with Graph() as g:
    n = 10000
    p = 10
    m = 5
    x = Variable(np.random.normal(0, 1, (n, p)))
    y = Variable(np.random.normal(0, 1, (n, m)))

    layer1 = Dense(output_shape=10, activation=sigmoid)
    layer2 = Dense(output_shape=m, activation=sigmoid)
    optimizer = GradientDescent(0.8)
    loss = mean_squared_error

    loss_array = []

    for i in range(300):
        if i % 50 == 0:
            print(i)
        a = layer1.forward(x)
        b = layer2.forward(a)
        c = loss(y, b)
        g.update_gradient_with_optimizer(c, optimizer)

        loss_array.append(c.value)

plt.plot(loss_array)
plt.show()
