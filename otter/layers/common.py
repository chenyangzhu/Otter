import numpy as np
from ..dam.structure import Variable


class Layer():
    def __init__(self):
        pass

    def train_forward(self, X):

        pass

    def predict_forward(self, X):

        return self.train_forward(X)

    @property
    def params(self):
        return 0


class Dense(Layer):
    def __init__(self, output_shape, activation, trainable=True):

        """
        :param output_shape:   # of hidden units
        :param activation:    an activation function
        """

        super().__init__()
        # Normal initialization
        self.m = output_shape
        self.trainable = trainable
        self.activation = activation
        self.initialize = True

    def train_forward(self, x: Variable):
        # Forward Propagation
        # print(x.shape)
        self.p = x.shape[1]  # input_tensor shape

        if self.initialize:
            self.w = Variable(np.random.normal(0, 1, (self.p, self.m)),
                              trainable=self.trainable, name='Dense_w')
            self.b = Variable(np.random.normal(0, 1, (self.m, 1)),
                              trainable=self.trainable, param_share=True,
                              name='Dense_b')
            self.initialize = False

        output = x.dot(self.w) + self.b.T()
        return self.activation(output)

    def predict_forward(self, x: Variable):
        return self.train_forward(x)


class Dropout(Layer):
    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate

    def train_forward(self, x: Variable):

        """
        The key of dropout, is reduce the dropout process to
        multiplying a random mask (0, 1).
        We can then simply use the standard grad-back-prop methods in the library.

        :param x:  for whichever dimension
        :return:
        """

        mask = np.ones(x.shape)
        dimension = len(x.shape)
        total_param = np.prod(x.shape)

        number_to_drop = np.int(total_param * self.dropout_rate)
        drop_coord = np.zeros((number_to_drop, dimension), dtype=np.int)

        for i in range(dimension):
            drop_coord[:, i] = np.random.randint(0, x.shape[i] - 1, number_to_drop).astype(int)

        for coord in drop_coord:

            mask[coord] = 0

        output = x.multiply(Variable(mask))

        return output

    def predict_forward(self, x: Variable):

        return x


class BatchNormalization(Layer):
    def __init__(self):
        super().__init__()

    def train_forward(self, x: Variable):
        x_val = x.value
        self.mean = Variable(np.average(x.value))
        self.var_inv = Variable(np.array(np.var(x_val))).safe_inv()
        output = x.sub(self.mean).multiply(self.var_inv)
        return output

    def predict_forward(self, x: Variable):

        output = x.sub(self.mean).multiply(self.var_inv)
        return output
