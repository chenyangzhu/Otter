import numpy as np
from otter._hyperparam import *
import otter


class Variable:

    def __init__(self, x, dtype=None, lchild=None, rchild=None,
                 trainable=False, param_share=False,
                 name=None):
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

        # if dtype is None, we automatically generate dtype from x
        # Else we check if the declared datatype is the same as the input data x
        if dtype is None:
            self.dtype = x.dtype
        else:
            assert x.dtype == dtype

        self.lchild = lchild
        self.rchild = rchild
        self.parent = None
        self.gradient = np.ones(self.value.shape)
        self.trainable = trainable
        self.param_share = param_share
        self.name = name
        self.back_prop = None

        # The variable is utilized to take notice of
        self.first_optimize = True

        '''
        The back_prop function is what we use to do back_propagation
        The function only exists when the Variable is derived from
        an input Variable.
        Thus, if there's no back_prop function, it must be the input
        and we no longer need to do the back_prop on it.
        '''

        self.back_prop = None

    def __str__(self):
        return "otter.Variable: " + str(self.name) + " " + str(self.dtype) + " (" + self.value.__str__() + ")"

    '''
    The followings are two basic functions we provide to allow faster 
    computations and extra abilities.
    '''

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

    def T(self):  # Checked
        self.parent = Variable(x=self.value.T,
                               lchild=self)

        self.parent.back_prop = self.parent.back_T
        return self.parent

    def back_T(self):  # Checked
        self.lchild.gradient = self.gradient.T

    def maximum(self, axis=None):

        if axis is not None:
            # Local Maximum, return an array
            output_shape = list(self.shape)
            output_shape[axis] = 1

            self.parent = Variable(np.max(self.value, axis=axis).reshape(output_shape),
                                   lchild=self)

            max_idx = np.argmax(self.value, axis=axis)
            mask = np.zeros(self.value.shape)

            # TODO Now we'll only do 2D cases
            if axis == 1:
                mask[np.arange(len(mask)), max_idx] = 1
            elif axis == 0:
                mask[max_idx, np.arange(mask.shape[1])] = 1

        else:
            # Global Maximum, return a scalar
            self.parent = Variable(np.max(self.value).reshape(()), lchild=self)
            mask = (self.parent.value == self.value).astype(int)

        self.parent.maximum_grad_parser = {'mask': mask}
        self.parent.back_prop = self.parent.back_maximum

        # self.parent = Variable(np.max(self.value),
        #                        lchild=self,
        #                        path='maximum')

        # Method 1, only one max value
        # max_idx = np.argmax(self.value)
        # mask = np.zeros(self.value.shape)
        # mask[max_idx] = 1

        # Method 2, can have multiple maximum

        return self.parent

    def back_maximum(self):
        self.lchild.gradient = np.multiply(self.gradient,
                                           self.maximum_grad_parser['mask'])

    def safe_exp(self):
        return self.clip(-VALUE_CLIPPING_THRESHOLD,
                         VALUE_CLIPPING_THRESHOLD).exp()

    def exp(self):
        self.parent = Variable(np.exp(self.value),
                               lchild=self)
        self.parent.back_prop = self.parent.back_exp
        return self.parent

    def back_exp(self):
        self.lchild.gradient = np.multiply(self.gradient, self.value)

    def safe_inv(self):
        return (self + Variable(np.array(EPSILON))).inv()

    def inv(self):

        # In case of 1/0
        # self.value += 0.0001

        self.parent = Variable(1 / self.value, lchild=self)
        self.parent.back_prop = self.parent.back_inv
        return self.parent

    def back_inv(self):
        self.lchild.gradient = - np.multiply(self.gradient, self.value ** 2)

    def neg(self):
        self.parent = Variable(- self.value,
                               lchild=self)
        self.parent.back_prop = self.parent.back_neg
        return self.parent

    def back_neg(self):
        self.lchild.gradient = - self.gradient

    def sum(self, axis=None):
        """
        :param axis: if axis = None, return global sum
        :return:
        """
        if axis is not None:
            '''
            When axis is not None, by default the sum function in numpy will return
            a shape probably (1,) for 2-dim matrix. This is true, because all vectors should be a column vector.
            
            We'll be good in forward propagation, so we will leave the shape unchanged.
            However, this will be very dangerous for later back-prop, so we need to reshape
            the coming gradients to the correct form. This could be seen in later back-prop functions.
            '''
            # Finding the correct shape
            # Notice that whenever our axis is, the output shape should have one on the corresponding axis,
            # and the original shape on other axis's.
            output_shape = list(self.shape)
            output_shape[axis] = 1
            # print(output_shape)

            self.parent = Variable(np.sum(self.value, axis=axis).reshape(output_shape),
                                   lchild=self)
            # print(self.parent.shape)
            # self.parent = Variable(np.sum(self.value, axis=axis),
            #                        lchild=self, path='sum')

        else:
            '''
            When axis is None, the value returns to a scalar. While we keep it as a Variable type,
            it should be a better case to have it as a () dimension
            '''
            self.parent = Variable(np.sum(self.value).reshape(()),
                                   lchild=self)

        self.parent.sum_grad_parser = {"axis": axis,
                                       "shape": self.shape}
        self.parent.back_prop = self.parent.back_sum
        return self.parent

    def back_sum(self):
        axis = self.sum_grad_parser['axis']
        shape = self.sum_grad_parser['shape']
        # TODO Notice that this is just a 2D case
        if axis is None:
            self.lchild.gradient = np.ones(shape) * self.gradient
        else:
            # We need to reshape gradients
            gradient_shape = list(shape)  # n, m
            # print(self.gradient.shape)
            self.lchild.gradient = self.gradient.repeat(gradient_shape[axis], axis=axis)

    def __pow__(self, power, modulo=None):
        return self.pow(power)

    def pow(self, power):

        self.parent = Variable(self.value ** power,
                               lchild=self)
        self.parent.pow_grad_parser = {'power': power,
                                       'value': self.value}
        self.parent.back_prop = self.parent.back_pow
        return self.parent

    def back_pow(self):
        power = self.pow_grad_parser['power']
        value = self.pow_grad_parser['value']
        self.lchild.gradient = np.multiply(power * value ** (power - 1),
                                           self.gradient)

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
                                   lchild=self)

        else:
            self.parent = Variable(np.average(self.value), lchild=self)

        self.parent.average_grad_parser = {"axis": axis,
                                           "shape": self.shape}
        self.parent.back_prop = self.parent.back_average
        return self.parent

    def back_average(self):
        axis = self.average_grad_parser['axis']
        shape = self.average_grad_parser['shape']

        if axis is None:
            self.lchild.gradient = np.ones(shape) * self.gradient / np.prod(shape)
        else:
            # We need to reshape gradients
            gradient_shape = list(shape)
            gradient_shape[axis] = 1
            # print(self.gradient.shape)
            # print(shape)
            # print(gradient_shape)
            self.gradient = self.gradient.reshape(gradient_shape)
            self.lchild.gradient = np.ones(shape) * self.gradient / shape[axis]

    def safe_log(self):
        return (self + Variable(np.array(EPSILON))).log()

    def log(self):
        self.parent = Variable(np.log(self.value),
                               lchild=self)
        # OVERFLOW PREVENTION: TO prevent 1 / 0 cases
        inverse = 1 / self.value

        self.parent.log_grad_parser = {"inverse": inverse}
        self.parent.back_prop = self.parent.back_log
        return self.parent

    def back_log(self):
        self.lchild.gradient = np.multiply(self.log_grad_parser['inverse'],
                                           self.gradient)

    def slice(self, index, axis):
        """
        This function selects the index from the matrix
        :param index: !! must be a 1d array!!!!! cannot be a matrix!!!!!
        :param axis:
        :return:
        """

        # TODO This is a 2D example
        mask = np.zeros_like(self.value)
        if axis == 0:
            output = self.value[index,
                                np.arange(self.shape[1])].reshape((1, self.shape[1]))
            mask[index, np.arange(self.shape[1])] = 1
        elif axis == 1:
            output = self.value[np.arange(self.shape[0]), index].reshape((self.shape[0], 1))
            mask[np.arange(self.shape[0]), index] = 1
            # print(index)

        self.parent = Variable(output, lchild=self)
        self.parent.slice_grad_parser = {"mask": mask}
        self.parent.back_prop = self.parent.back_slice
        return self.parent

    def back_slice(self):
        self.lchild.gradient = np.multiply(self.slice_grad_parser['mask'], self.gradient)

    def tanh(self):
        self.parent = Variable((np.exp(self.value) - np.exp(-self.value)) / (np.exp(self.value) + np.exp(-self.value)),
                               lchild=self)
        self.parent.back_prop = self.parent.back_tanh
        return self.parent

    def back_tanh(self):
        self.lchild.gradient = np.multiply(1 - self.lchild.value ** 2, self.gradient)

    def clip(self, floor, ceiling):
        """
        Value clipping
        :param floor:
        :param ceiling:
        :return:
        """
        self.parent = Variable(np.clip(self.value, floor, ceiling),
                               lchild=self)
        self.parent.back_prop = self.parent.back_clip
        return self.parent

    def back_clip(self):
        # Do not change the gradient, directly pass on.
        self.lchild.gradient = self.gradient

    # @timer
    def reshape(self, new_shape):
        self.parent = Variable(self.value.reshape(new_shape),
                               lchild=self)
        self.parent.back_prop = self.parent.back_reshape
        return self.parent

    def back_reshape(self):
        self.lchild.gradient = self.gradient.reshape(self.lchild.shape)

    '''
    These functions require two inputs
    '''

    def __add__(self, y):
        return self.add(y)

    def add(self, y):
        self.parent = Variable(self.value + y.value,
                               lchild=self, rchild=y)
        self.parent.back_prop = self.parent.back_add
        return self.parent

    def back_add(self):
        self.lchild.gradient = self.gradient
        self.rchild.gradient = self.gradient

    def __sub__(self, y):
        return self.sub(y)

    def sub(self, y):

        self.parent = Variable(self.value - y.value,
                               lchild=self, rchild=y)
        self.parent.back_prop = self.parent.back_sub
        return self.parent

    def back_sub(self):
        self.lchild.gradient = self.gradient
        self.rchild.gradient = -self.gradient

    def dot(self, y):
        """
        self: n x p
        y:    p x m
        output: n x m
        grad_self: grad_output * y.T
        grad_y:    self.T * grad_output
        """
        self.parent = Variable(np.dot(self.value, y.value),
                               lchild=self, rchild=y)
        self.parent.dot_grad_parser = {"x": self.value,
                                       "y": y.value}
        self.parent.back_prop = self.parent.back_dot
        return self.parent

    def back_dot(self):
        self.lchild.gradient = np.matmul(self.gradient,
                                         self.dot_grad_parser['y'].T)
        self.rchild.gradient = np.matmul(self.dot_grad_parser['x'].T,
                                         self.gradient)

    def multiply(self, y):
        """Element-wise multiplication

        Notice very strictly that this multiplication must have shape of self
        equals to shape of y
        """

        assert self.shape == y.shape

        # Normal Process
        self.parent = Variable(np.multiply(self.value, y.value),
                               lchild=self, rchild=y)
        self.parent.multiply_grad_parser = {"x": self.value,
                                            "y": y.value}
        self.parent.back_prop = self.parent.back_multiply
        return self.parent

    def back_multiply(self):
        self.lchild.gradient = np.multiply(self.gradient,
                                           self.multiply_grad_parser['y'])
        self.rchild.gradient = np.multiply(self.gradient,
                                           self.multiply_grad_parser['x'])

    def repeat(self, repeat_number, axis):

        self.parent = Variable(np.repeat(self.value, repeat_number, axis),
                               lchild=self)
        self.parent.back_prop = self.parent.back_repeat
        self.parent.repeat_axis = axis

        return self.parent

    def back_repeat(self):
        self.lchild.gradient = np.average(self.gradient, axis=self.repeat_axis).reshape(self.lchild.gradient.shape)

    """
    Activations
    """

    def back_softmax(self):
        # First create the eye matrix
        n, m = self.shape

        eyed = self.value.reshape(n, m, 1) * np.eye(m)

        ones = self.value.reshape(n, m, 1) * np.ones(m)
        inverse_ones = self.value.reshape(n, 1, m) * np.ones((m, m))

        ds = eyed - np.multiply(ones, inverse_ones)
        avg_ds = np.average(ds, axis=0)

        self.lchild.gradient = np.matmul(self.gradient, avg_ds)

    """
    Layers
    """

    def back_flatten(self):
        # print(self.gradient.reshape(self.flatten_grad_parser['shape']))
        self.lchild.gradient = self.gradient.reshape(self.flatten_grad_parser['shape'])

    def back_maxpooling2d(self):
        n, c, x_new, y_new = self.size

        grad_x = np.zeros((n, c, x_new, y_new))

        for image_idx in range(n):
            for channel_idx in range(c):
                for i in range(x_new):
                    for j in range(y_new):
                        grad_x[image_idx, channel_idx][tuple(self.mapping[i, j])] = self.gradient[
                            image_idx, channel_idx, i, j]
        self.lchild.gradient = grad_x

    def sparse_dot_with_mapping(self, w, mapping, sparse_matrix_height, sparse_matrix_width):
        """
        This function calculates a sparse matrix multiplication of xw

        Args:

            x is a dense matrix, stored as a type Variable
            w is a sparse matrix, but stored in a dense way, as a type Variable
            mapping is the mapping from w to the *fake* sparse matrix which is a list
                    containing the index for [(w), (fake sparse matrix)]
        """

        output = otter.zeros(shape=(self.shape[0], sparse_matrix_width), dtype=np.float32)
        output.rchild = w
        output.lchild = self
        output.back_prop = output.back_sparse_dot_with_mapping

        for each_mapping in mapping:
            index_in_w = each_mapping[0]
            i, j = each_mapping[1]

            # the i,j th element in the sparse matrix,
            # is multiplied by the ith column from the x matrix
            # and is added to jth column in the output column.
            # print("self.value", self.value[:, i])
            # print("w value", w.value[index_in_w])

            output.value[:, j] += self.value[:, i] * w.value[index_in_w]

        output.sparse_dot_with_mapping_grad_parser = {'mapping': mapping}

        return output

    def back_sparse_dot_with_mapping(self):
        """
        This function is the back_propagation of the sparse matrix multiplication
        defined above
        """
        mapping = self.sparse_dot_with_mapping_grad_parser['mapping']
        # print(self.rchild)
        # print(self.lchild)

        self.rchild.gradient = np.zeros_like(self.rchild.gradient)
        self.lchild.gradient = np.zeros_like(self.lchild.gradient)

        for each_mapping in mapping:

            index_in_w = each_mapping[0]
            i, j = each_mapping[1]
            self.rchild.gradient[index_in_w] += np.sum(self.lchild.value[:, i] * self.gradient[:, j])
            self.lchild.gradient[:, i] += self.rchild.value[index_in_w] * self.gradient[:, j]