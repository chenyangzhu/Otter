import numpy as np
from nn import *


class Loss:
    def __init__(self):
        pass

    def forward(self, y, yhat):
        self.output = 0
        return self.output


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


class categorical_crossentropy(Loss):

    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        # TODO 修改下方
        self.y = y
        self.yhat = yhat
        # print(y.shape)
        # print(yhat.shape)
        assert y.shape == yhat.shape
        print(y * np.log(yhat) + (1-y) * np.log(1 - yhat))
        self.output = - np.sum(y * np.log(yhat) + (1-y) * np.log(1 - yhat)) / y.shape[0]
        return self.output

    @property
    def gradient(self):
        return (self.yhat - self.y) / self.y.shape[0]

    @property
    def loss(self):
        return self.output