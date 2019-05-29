import matplotlib.pyplot as plt

from klausnet import layers
from klausnet.loss import mean_squared_error, categorical_crossentropy
from klausnet.optimizer import GradientDescent
from klausnet.activation import Sigmoid, Linear, Softmax, Relu
from klausnet.nn import *
from klausnet.model import Sequential

np.random.seed(2019)

n = 100
channel = 3
x_row = 64
x_col = 64

X = np.random.normal(0, 1, (n, channel, x_row, x_col))
y = np.random.normal(0, 1, (n, 1))

model = Sequential([layers.Conv2D(input_shape=X.shape[1:],
                                  filters=3,
                                  kernel_size=(3,3),
                                  strides=(1,1),
                                  padding=(0,0),
                                  activation=Sigmoid())])

model.compile(optimizer=GradientDescent(learning_rate=0.1),
              loss=mean_squared_error(),
              iteration=100)

history = model.fit(X, y)
