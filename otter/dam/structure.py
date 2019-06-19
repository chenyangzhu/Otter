import numpy as np
VALUE_CLIPPING_THRESHOLD = 2e2

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
        return (self + Variable(np.ones(1) * 1e-6)).inv()

    def inv(self):

        # In case of 1/0
        # self.value += 0.0001

        self.parent = Variable(1 / self.value,
                               lchild=self)
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
                                   lchild=self)
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

        if axis is None:
            self.lchild.gradient = np.ones(shape) * self.gradient
        else:
            # We need to reshape gradients
            gradient_shape = list(shape)
            gradient_shape[axis] = 1
            self.gradient = self.gradient.reshape(gradient_shape)

            self.lchild.gradient = np.ones(shape) * self.gradient

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
            self.parent = Variable(np.average(self.value).reshape(()), lchild=self)

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
        return (self + Variable(np.ones(1) * 1e-6)).log()

    def log(self):
        # self.value += 0.0001

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
        # print("+++++++++++++++back_slice++++++++++++++++++")
        # print(self.slice_grad_parser['mask'])
        # print(np.multiply(self.slice_grad_parser['mask'], self.gradient))
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
        self.parent = Variable(np.matmul(self.value, y.value),
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
        """
        Element-wise multiplication
        :param y:
        :return:
        """
        # print(self.value)
        '''
        Value Clipping
        '''

        # self.value = np.multiply(self.value, self.value < VALUE_CLIPPING_THRESHOLD)
        # self.value = np.multiply(self.value, self.value > -VALUE_CLIPPING_THRESHOLD)

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

    # def div(self, y):
    #     """
    #     Element-wise division
    #     :param y:
    #     :return:
    #     """
    #     self.parent = Variable(self.value / y.value,
    #                            lchild=self, rchild=y)
    #     self.parent.back_prop = self.parent.back_div
    #     return self.parent
    #
    # def back_div(self):
    #     # TODO write gradients
    #     self.lchild.gradient = 0

    """
    Activations
    """

    def back_softmax(self):
        self.lchild.gradient = np.multiply(self.value * (1 - self.value),
                                           self.gradient)

    # def back_tanh(self):
    #     M = self.tanh_grad_parser['M']
    #     xvalue = self.tanh_grad_parser['xvalue']
    #     self.lchild.gradient = (np.exp(xvalue-M) + np.exp(-xvalue - M) - np.exp(-M))/(np.exp(xvalue-M) - np.exp(-xvalue-M))

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

    def back_mapping(self):
        for each in self.w2mapping:
            self.lchild.gradient[each[0]] = self.gradient[each[1]]

    def conv2d(self, w, stride, padding):
        """
        Forward propagation of convolution layer
        :param w:
        :param stride:
        :param padding:
        :return:
        """
        f, c, u, v = w.shape        # Kernel size
        n, c, x, y = self.shape     # Input shape
        kernel_size = [u, v]

        x_new = int((x - kernel_size[0] + 2 * padding[0]) / stride[0] + 1)
        y_new = int((y - kernel_size[1] + 2 * padding[1]) / stride[1] + 1)

        self.parent = Variable(np.zeros((n, f, x_new, y_new)),
                               lchild=self, rchild=w)

        '''
        In this mapping matrix, we use lists to store a 3d mapping with
        [new_matrix_index, old_matrix_index, and weight_matrix_index]
        During each iteration, whenever we have a mappping, we append a new list
        containing these three variables.

        During the back_prop, we search in the list for the corresponding index,
        and sum all the rows we have.
        '''

        # Method 2
        # We first create a mapping
        mapping = []
        # def idx_four2two(idx, shape):
        #     """
        #     This function transform a 4d index (n, channel, x, y) to a 1d number,
        #     with accordance to np.reshape()
        #
        #     :param idx:     the index
        #     :param shape:   the input_tensor shape [n, max_channel, max_x, max_y]
        #     :return:
        #     """
        #     new_idx = idx[1] * np.prod(shape[2:]) + idx[2] * shape[3] + idx[3]
        #     return idx[0], new_idx

        # def idx_three2one(idx, shape):
        #     new_idx = idx[0] * np.prod(shape[1:]) + idx[1] * shape[2] + idx[2]
        #     return new_idx

        # Building the mapping
        for filter_idx in range(f):
            for i in range(x_new):
                for j in range(y_new):
                    x_start = int(i * stride[0])
                    y_start = int(j * stride[1])
                    for ix in range(u):
                        for jx in range(v):
                            for channel_idx in range(c):
                                mapping.append([(filter_idx, i, j),
                                                (channel_idx, x_start + ix, y_start + jx),
                                                (filter_idx, channel_idx, ix, jx)])
        # Now mapping is already a valid sparse matrix with weight on its coords



        # We first need to reshape our x matrix
        # input_image_flattened = self.reshape((n, c*x*y))
        # new_image_flattened = input_image_flattened.dot(mapping)
        # new_image = new_image_flattened.reshape((n, f, x_new, y_new))
        # mapping = []

        record = True
        for image_idx in range(n):          # For each image
            for filter_idx in range(f):    # For each channel
                # The mapping for each image and each channel is the same,
                # so we only need to record the mapping once.
                for i in range(x_new):
                    for j in range(y_new):  # For each in the coordinates.

                        # First, find the clip coordinate
                        x_start = int(i * stride[0])
                        x_end = int(x_start + u)
                        y_start = int(j * stride[1])
                        y_end = int(y_start + v)

                        # Record the mapping for later back-prop
                        if record:
                            for ix in range(u):
                                for jx in range(v):  # For each element in the weight matrix
                                    for channel_idx in range(c):
                                        mapping.append([(filter_idx, i, j),
                                                        (channel_idx, x_start + ix, y_start + jx),
                                                        (filter_idx, channel_idx, ix, jx)])
                        # end record

                        # Computation
                        clip = self.value[image_idx, :, x_start: x_end, y_start: y_end]
                        weighted_clip = np.multiply(clip, w.value[filter_idx])
                        sum_of_weighted_clip = np.sum(weighted_clip)

                        self.parent.value[image_idx, filter_idx, i, j] = sum_of_weighted_clip

                    # end for j
                # end for i
            # end for filter_idx
            record = False  # End record, cause others are the same.
        # end for image_idx
        self.parent.conv2d_grad_parser = {'x_shape': [n, c, x, y],
                                          'w_shape': [f, c, u, v],
                                          'w': w.value,
                                          'x': self.value,
                                          'mapping': mapping}
        self.parent.back_prop = self.parent.back_conv2d
        return self.parent

    def back_conv2d(self):
        n, c, x, y = x_shape = self.conv2d_grad_parser['x_shape']
        f, c, u, v = w_shape = self.conv2d_grad_parser['w_shape']
        mapping = self.conv2d_grad_parser['mapping']
        w = self.conv2d_grad_parser['w']
        X = self.conv2d_grad_parser['x']

        gradient_x = np.zeros(x_shape)  # n c x y
        gradient_w = np.zeros(w_shape)  # f c u v

        # Logic 2, iterate through all the rows in mapping matrix
        # We then directly update the gradients in gradient_x and gradient_w
        # print(self.gradient.shape)
        # print(x_shape)
        # print(w_shape)

        for each_map in mapping:
            # print(each_map)

            # In this iteration, we want to update the X[:, :, each_map[1]]
            # and we want to update the w[:, :, each_map[2]]
            # Despite the fact that the first dimension of X and w are different,
            # one is n and the other is f

            # We first update the gradient for x
            gradient_x[:, each_map[1][0], each_map[1][1], each_map[1][2]] = self.gradient[:, each_map[0][0], each_map[0][1], each_map[0][2]] * \
                                                                          w[each_map[2][0], each_map[2][1], each_map[2][2], each_map[2][3]]

            # We then want to update the gradient for w
            gradient_w[each_map[2][0], each_map[2][1], each_map[2][2], each_map[2][3]] = np.matmul(self.gradient[:, each_map[0][0], each_map[0][1], each_map[0][2]].T,
                                                                          X[:, each_map[1][0], each_map[1][1], each_map[1][2]])

        # Logic 1, the very stupid approach.
        for image_idx in range(n):
            for channel_idx in range(c):
                for i in range(x):
                    for j in range(y):  # for each_element in the gradient matrix

                        # search the mapping matrix for all (i,j) coord in the old matrix
                        # i.e. the second col.
                        output = 0

                        # The logic here, is we iterate through the mapping matrix,
                        # We then find all the rows that satisfies the index for the old x's.
                        #
                        for each_mapping in mapping:
                            if each_mapping[1] == (i, j):
                                # old x matrix, or grad_x or lchild gradient matrix
                                for filter_idx in range(f):
                                    output += self.gradient[image_idx, filter_idx][each_mapping[0]] \
                                              * w[filter_idx, channel_idx][each_mapping[2]]

                        gradient_x[image_idx, channel_idx, i, j] = output

                # End for each element in one channel of x
            # End for each channel
        # End for each image

        for filter_idx in range(f):
            for channel_idx in range(c):
                for i in range(u):
                    for j in range(v):  # for each_element in the gradient matrix

                        # search the mapping matrix for all (i,j) coord in the old matrix
                        # i.e. the second col.
                        output = 0
                        for each_mapping in mapping:

                            if each_mapping[2] == (i, j):

                                # old x matrix, or grad_x or lchild gradient matrix
                                for image_idx in range(n):
                                    output += self.gradient[image_idx, filter_idx][each_mapping[0]] \
                                              * X[image_idx, channel_idx][each_mapping[1]]

                        gradient_w[filter_idx, channel_idx, i, j] = output

                # End for each element in one channel of x
            # End for each channel
        # End for each image

        self.lchild.gradient = gradient_x
        self.rchild.gradient = gradient_w
