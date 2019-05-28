import matplotlib.pyplot as plt

from klausnet import layers
from klausnet.loss import mean_squared_error, categorical_crossentropy
from klausnet.optimizer import GradientDescent
from klausnet.activation import Sigmoid, Linear, Softmax
from klausnet.nn import *
from klausnet.model import Sequential
import klausnet.history as hist


np.random.seed(2029)

n = 100
p = 20
m = 4
X = np.random.normal(0, 1, (n, p))  # n x p
w = np.random.normal(0, 1, (p, m))  # p x m
b = np.random.normal(0, 1, (n, m))  # n x m
y = np.random.randint(0,3,(n,1))        # n x m
# print(y)

model = Sequential([layers.Dense(hidden_unit=128,
                                 input_shape=(n, p),
                                 activation=Sigmoid()),
                    layers.Dense(hidden_unit=m,
                                 input_shape=(n, 128),
                                 activation=Softmax())
                    ])

model.compile(optimizer=GradientDescent(learning_rate=0.1),
              loss=categorical_crossentropy(),
              iteration=200)

history = model.fit(X, y)

yhat = model.predict(X)

plt.plot(history['loss'])
plt.show()