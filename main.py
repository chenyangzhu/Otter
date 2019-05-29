import matplotlib.pyplot as plt

from klausnet import layers
from klausnet.loss import mean_squared_error, categorical_crossentropy
from klausnet.optimizer import GradientDescent
from klausnet.activation import Sigmoid, Linear, Softmax, Relu
from klausnet.nn import *
from klausnet.model import Sequential
import klausnet.history as hist


np.random.seed(2019)

n = 1000
p = 50
m = 10
X = np.random.normal(0, 1, (n, p))           # n x p
w = np.random.normal(0, 1, (p, m))           # p x m
b = np.random.normal(0, 1, (n, m))           # n x m
# y = np.random.randint(0, m-1, (n, 1))        # n x m
y = np.sign(np.matmul(X, w) + b)
# print(y)

model = Sequential([layers.Dense(hidden_unit=m,
                                 input_shape=(n, p),
                                 activation=Sigmoid()),
                    layers.Dense(hidden_unit=m,
                                 input_shape=(n, m),
                                 activation=Sigmoid()),
                    layers.Dense(hidden_unit=m,
                                 input_shape=(n, m),
                                 activation=Relu())
                    ])

model.compile(optimizer=GradientDescent(learning_rate=0.1),
              loss=mean_squared_error(),
              iteration=1000)

history = model.fit(X, y)

yhat = model.predict(X)

plt.plot(history['loss'])
plt.show()