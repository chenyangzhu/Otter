import numpy as np
import matplotlib.pyplot as plt

from otter import Variable
from otter.loss import mean_squared_error
from otter.dam.graph import Graph
from otter.activation import sigmoid
from otter.layers.common import Dense, Dropout, BatchNormalization
from otter.optimizer import GradientDescent, StochasticGradientDescent
from otter.model import Sequential
np.random.seed(2020)


with Graph() as g:
    n = 1000
    p = 10
    m = 1
    x = Variable(np.random.normal(0, 1, (n, p)))
    y = Variable(np.random.normal(0, 1, (n, m)))

    layer1 = Dense(output_shape=10,
                   input_shape=p,
                   activation=sigmoid)
    norm = BatchNormalization()
    layer2 = Dense(output_shape=1,
                   input_shape=10,
                   activation=sigmoid)

    model = Sequential([layer1, norm, layer2])

    optimizer = StochasticGradientDescent(0.1)
    loss = mean_squared_error

    model.compile(graph=g, optimizer=optimizer,
                  loss=loss, epoch=200, batch=10)

    history = model.fit(x, y)

plt.plot(history['loss'])
plt.show()

model.predict(x)