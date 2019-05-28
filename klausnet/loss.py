from klausnet.nn import *


class Loss:
    def __init__(self):
        pass

    def forward(self, y, yhat):
        self.output = 0
        return self.output


class mean_squared_error(Loss):
    """
    (y - yhat)^2
    """

    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        self.y = y
        self.yhat = yhat
        assert y.shape == yhat.shape

        self.output = np.sum((y - yhat) ** 2) / y.shape[0]
        return self.output

    @property
    def gradient(self):
        return np.multiply((self.y - self.yhat), self.y) * 2 / self.y.shape[0]

    @property
    def loss(self):
        return self.output


class categorical_crossentropy(Loss):
    """
    Notice 如果使用categorical_crossentropy,
    先前必须要使用softmax activation,变为概率
    """
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        '''

        :param y: categorical 属于 [0, 1, 2, 3, 4, ... , #of classes - 1]
        :param yhat: *已经经过softmax了的矩阵*: n sample x p classes
        :return:
        '''
        # TODO 修改下方
        self.y = y
        self.yhat = yhat
        print("yhat shape")
        print(yhat.shape)
        print(y.shape)
        # assert y.shape == yhat.shape

        # 首先，我们看哪些yhat分对了
        # 判别标准就是看每行的最大值的index，是不是category里对应的

        predicted_label = np.argmax(yhat, axis=1).reshape((yhat.shape[0],1))
        # print(predicted_label.shape)
        mask = (predicted_label == y).astype(int)
        print(mask)

        # 原来的矩阵乘上 mask 之后就只剩下我们需要的啦
        self.output = - np.sum(np.multiply(np.log(yhat), mask))
        print(self.output)
        return self.output

    @property
    def gradient(self):
        return (self.yhat - self.y) / self.y.shape[0]

    @property
    def loss(self):
        return self.output