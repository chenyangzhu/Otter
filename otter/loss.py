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
        assert y.shape == yhat.shape

        # Calculate train_forward prop
        self.output = np.sum((y - yhat) ** 2 / y.shape[0])

        # Calculate Gradients
        self.__gradient = np.multiply((y - yhat), y) * 2 / y.shape[0]

        return self.output

    @property
    def gradient(self):
        return self.__gradient

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
        # print("yhat shape")
        # print(yhat.shape)
        # print(y.shape)
        # assert y.shape == yhat.shape
        n = yhat.shape[0]
        # 首先，我们看哪些yhat分对了
        # 判别标准就是看每行的最大值的index，是不是category里对应的

        predicted_label_idx = np.argmax(yhat, axis=1).reshape((n, 1))
        predicted_label_prob = np.max(yhat, axis=1).reshape((n, 1))
        # print(predicted_label.shape)
        mask = (predicted_label_idx == y).astype(int)

        logged_predicted_label_prob = np.log(predicted_label_prob)
        logged_predicted_label_prob_with_mask = np.multiply(mask, logged_predicted_label_prob)

        self.output = - np.sum(logged_predicted_label_prob_with_mask) / n
        # print(mask)

        # 原来的矩阵乘上 mask 之后就只剩下我们需要的啦
        # self.output = - np.sum(np.multiply(np.log(yhat), mask))
        # print(self.output)

        # Calculate Gradients
        # 求 gradient 就是要把 max 位置变为 1 / 原来的数
        # 把其他位置全都变为0.
        # TODO 现在用了一个蠢办法
        labs = np.concatenate([np.array(range(n)).reshape((n, 1)),
                               predicted_label_idx], axis=1)
        gradient = np.zeros(yhat.shape)
        for label in labs:
            # print(label)
            gradient[label] = 1 / yhat[label]
        gradient = np.multiply(gradient, mask)

        self.__gradient = - gradient

        return self.output

    @property
    def gradient(self):
        return self.__gradient

    @property
    def loss(self):
        return self.output