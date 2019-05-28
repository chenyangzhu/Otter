import numpy as np
from nn import *


class Loss:
    def __init__(self):
        pass

    def forward(self, y, yhat):
        pass

    @property
    def gradient(self):
        return 0


class mean_squared_error(Loss):
    '''
    (y-yhat)^2
    '''

    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        self.y = y
        self.yhat = yhat
        # print(y.shape)
        # print(yhat.shape)
        assert y.shape == yhat.shape

        self.output = np.sum((y - yhat) ** 2)
        return self.output

    @property
    def gradient(self):
        return np.multiply((self.y - self.yhat), self.y) * 2

    @property
    def loss(self):
        return self.output