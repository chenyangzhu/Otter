import numpy as np
import matplotlib.pyplot as plt

from otter import Variable
from otter.loss import mean_squared_error
from otter.dam.graph import Graph
from otter.activation import sigmoid
from otter.layers.common import Dense, Dropout
from otter.optimizer import GradientDescent

np.random.seed(2019)


with Graph() as g:
    n = 1000
    p = 10
    m = 1
    x = Variable(np.random.normal(0, 1, (n, p)))
    y = Variable(np.random.normal(0, 1, (n, m)))

    layer1 = Dense(output_shape=10, activation=sigmoid)
    layer2 = Dense(output_shape=m, activation=sigmoid)
    optimizer = GradientDescent(0.8)
    loss = mean_squared_error

    loss_array = []

    for _ in range(1000):
        a = layer1.train_forward(x)
        b = layer2.train_forward(a)
        c = loss(y, b)
        print(layer1.w.gradient)
        g.update_gradient_with_optimizer(c, optimizer)

        loss_array.append(c.value)

plt.plot(loss_array)
plt.show()
