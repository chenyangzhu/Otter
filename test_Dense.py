import matplotlib.pyplot as plt

from klausnet import layers
from klausnet.loss import mean_squared_error, categorical_crossentropy
from klausnet.optimizer import GradientDescent
from klausnet.activation import Sigmoid, Linear, Softmax, Relu
from klausnet.nn import *
from klausnet.model import Sequential

np.random.seed(2019)

n = 2000
p = 256
m = 10
X = np.random.normal(0, 1, (n, p))           # n x p
w = np.random.normal(0, 1, (p, m))           # p x m
b = np.random.normal(0, 1, (n, m))           # n x m
# y = np.random.randint(0, m-1, (n, 1))        # n x m
y = np.matmul(X, w) + b
# print(y)

model = Sequential([layers.Dense(hidden_unit=128,
                                 input_shape=(n, p),
                                 activation=Sigmoid()),
                    layers.Dense(hidden_unit=m,
                                 input_shape=(n, 128),
                                 activation=Sigmoid()),
                    layers.Dense(hidden_unit=m,
                                 input_shape=(n, m),
                                 activation=Relu())
                    ])

model.compile(optimizer=GradientDescent(learning_rate=0.1),
              loss=mean_squared_error(),
              iteration=100,
              batch=16,
              gradient_method='stochastic',  # 需要 gradient method 是配合使用不同的 optimization method
              minibatch=-1  # -1 就是不要mini batch
              )

history = model.fit(X, y)

yhat = model.predict(X)

plt.plot(history['loss'])
plt.show()