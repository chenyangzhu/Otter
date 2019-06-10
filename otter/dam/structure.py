import numpy as np


class Variable:

    def __init__(self, x, lchild=None, rchild=None, path='input',
                 trainable=False, param_share=False):
        """
        :param x:
        :param lchild:
        :param rchild:
        :param path:  To record the path where this Variable is derived.
                      Notice that the paths here are by default 'input'
        :param trainable:
        :param param_share: Does this variable contains 0 that we need to average.
        """

        self.value = x
        self.lchild = lchild
        self.rchild = rchild
        self.parent = None
        self.gradient = np.ones(self.value.shape)
        self.path = path
        self.trainable = trainable
        self.param_share = param_share

    def __str__(self):
        return "otter.Variable(" + self.value.__str__() + ")"

    @property
    def shape(self):
        return self.value.shape

    def detach(self):
        """
        Sometimes, we don't really want this Variable to record any previous data,
        so we can use detach, to detach it from all previous nodes.
        """
        self.lchild = None
        self.rchild = None

    '''
    These methods are only a transformation of the variable itself. They do not need any parent nodes.
    They are registered as lchild.
    '''

    def T(self):
        self.parent = Variable(x=self.value.T,
                               lchild=self,
                               path='T')
        return self.parent

    def maximum(self):
        self.parent = Variable(np.max(self.value),
                               lchild=self,
                               path='maximum')

        # Method 1, only one max value
        # max_idx = np.argmax(self.value)
        # mask = np.zeros(self.value.shape)
        # mask[max_idx] = 1

        # Method 2, can have multiple maximum
        mask = (self.parent.value == self.value).astype(int)

        self.parent.maximum_grad_parser = {'mask': mask}

        return self.parent

    def exp(self):
        self.parent = Variable(np.exp(self.value),
                               lchild=self,
                               path='exp')
        return self.parent

    def inv(self):
        self.parent = Variable(1 / self.value,
                               lchild=self,
                               path='inv')
        return self.parent

    def neg(self):
        self.parent = Variable(- self.value,
                               lchild=self,
                               path='neg')
        return self.parent

    def sum(self, axis=None):
        """
        :param axis: if axis = None, return global sum
        :return:
        """
        if axis is not None:
            '''
            When axis is not None, by default the sum function in numpy will return
            a shape probabily (1,) for 2-dim matrix. This is true, because all vectors should be a column vector.
            
            We'll be good in forward propagation, so we will leave the shape unchanged.
            However, this will be very dangerous for later back-prop, so we need to reshape
            the coming gradients to the correct form. This could be seen in later back-prop functions.
            '''
            # Finding the correct shape
            # Notice that whenever our axis is, the output shape should have one on the corresponding axis,
            # and the original shape on other axis's.
            output_shape = list(self.shape)
            output_shape[axis] = 1

            self.parent = Variable(np.sum(self.value, axis=axis).reshape(output_shape),
                                   lchild=self, path='sum')
            # self.parent = Variable(np.sum(self.value, axis=axis),
            #                        lchild=self, path='sum')

        else:
            '''
            When axis is None, the value returns to a scalar. While we keep it as a Variable type,
            it should be a better case to have it as a () dimension
            '''
            self.parent = Variable(np.sum(self.value).reshape(()), lchild=self, path='sum')

        self.parent.sum_grad_parser = {"axis": axis,
                                       "shape": self.shape}
        return self.parent

    def pow(self, power):

        self.parent = Variable(self.value ** power,
                               lchild=self, path='pow')
        self.parent.pow_grad_parser = {'power':power}

        return self.parent

    def average(self, axis=None):
        """
        This function is similar to sum.

        :param axis: if axis = None, return global sum
        :return:
        """
        if axis is not None:
            output_shape = list(self.shape)
            output_shape[axis] = 1

            self.parent = Variable(np.average(self.value, axis=axis).reshape(output_shape),
                                   lchild=self, path='average')

        else:
            self.parent = Variable(np.average(self.value).reshape(()), lchild=self, path='average')

        self.parent.average_grad_parser = {"axis": axis,
                                       "shape": self.shape}
        return self.parent

    '''
    These functions require two inputs
    '''

    def add(self, y):

        self.parent = Variable(self.value + y.value,
                               lchild=self, rchild=y,
                               path='add')
        return self.parent

    def sub(self, y):

        self.parent = Variable(self.value - y.value,
                               lchild=self, rchild=y,
                               path='sub')
        return self.parent

    def dot(self, y):
        """
        self: n x p
        y:    p x m
        output: n x m
        grad_self: grad_output * y.T
        grad_y:    self.T * grad_output
        :param y:
        :return:
        """
        self.parent = Variable(np.matmul(self.value, y.value),
                               lchild=self, rchild=y,
                               path="dot")
        self.parent.dot_grad_parser = {"x": self.value,
                                       "y": y.value}
        return self.parent

    def multiply(self, y):
        """
        Element-wise multiplication
        :param y:
        :return:
        """
        self.parent = Variable(np.multiply(self.value, y.value),
                               lchild=self, rchild=y,
                               path="multiply")
        self.parent.multiply_grad_parser = {"x": self.value,
                                            "y": y.value}
        return self.parent

    def div(self, y):
        """
        Element-wise division
        :param y:
        :return:
        """

        self.parent = Variable(self.value / y.value,
                               lchild=self, rchild=y,
                               path='div')
        return self.parent

    '''
    Update lchild and rchild gradients
    '''

    def update_lchild_gradient(self):
        """
        Every backprop, we need to update child nodes.
        :return:
        """
        # For self computations
        if self.path == "T":
            self.lchild.gradient = self.gradient.T
        elif self.path == "maximum":
            self.lchild.gradient = np.multiply(self.gradient, self.maximum_grad_parser['mask'])
        elif self.path == "exp":
            self.lchild.gradient = np.multiply(self.gradient, self.value)
        elif self.path == "inv":
            self.lchild.gradient = - np.multiply(self.gradient, self.value ** 2)
        elif self.path == "neg":
            self.lchild.gradient = - np.multiply(self.gradient, self.value)

        elif self.path == "sum":
            axis = self.sum_grad_parser['axis']
            shape = self.sum_grad_parser['shape']

            if axis is None:
                self.lchild.gradient = np.ones(shape) * self.gradient
            else:
                # We need to reshape gradients
                gradient_shape = list(shape)
                gradient_shape[axis] = 1
                self.gradient = self.gradient.reshape(gradient_shape)

                self.lchild.gradient = np.ones(shape) * self.gradient

        elif self.path == 'pow':
            power = self.pow_grad_parser['power']
            self.lchild.gradient = np.multiply(power * self.value ** (power - 1), self.gradient)

        elif self.path == "average":
            axis = self.average_grad_parser['axis']
            shape = self.average_grad_parser['shape']

            if axis is None:
                self.lchild.gradient = np.ones(shape) * self.gradient / np.prod(shape)
            else:
                # We need to reshape gradients
                gradient_shape = list(shape)
                gradient_shape[axis] = 1
                self.gradient = self.gradient.reshape(gradient_shape)

                self.lchild.gradient = np.ones(shape) * self.gradient / shape[axis]

        # For computations
        elif self.path == "add":
            self.lchild.gradient = self.gradient
        elif self.path == "sub":
            self.lchild.gradient = self.gradient
        elif self.path == "dot":
            self.lchild.gradient = np.matmul(self.gradient,
                                             self.dot_grad_parser['y'].T)
        elif self.path == "multiply":
            self.lchild.gradient = np.multiply(self.gradient,
                                               self.multiply_grad_parser['y'])
        elif self.path == "div":
            # TODO write gradients
            self.lchild.gradient = 0

        # For activations
        elif self.path == 'softmax':
            output_value = self.softmax_grad_parser["output"]
            self.lchild.gradient = np.multiply(output_value * (1 - output_value), self.gradient)


    def update_rchild_gradient(self):

        """
        Every backprop, we need to update child nodes.
        :return:
        """

        if self.path == "add":
            self.rchild.gradient = self.gradient
        elif self.path == "sub":
            self.rchild.gradient = - self.gradient
        # elif self.path == "T":  ## For transpose, you don't have an rchild.
        # self.rchild.gradient = self.gradient.T
        elif self.path == "dot":
            self.rchild.gradient = np.matmul(self.dot_grad_parser['x'].T,
                                             self.gradient)
        elif self.path == "multiply":
            self.rchild.gradient = np.multiply(self.gradient,
                                               self.multiply_grad_parser['x'])
