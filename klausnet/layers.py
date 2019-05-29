"""
Layers
- Dense
- CNN
- RNN

### 目前，Dense Layer是自带一个activation的！！！！！
但我们也可以把activation 作为一个新的layer，直接放到model里。
"""

from klausnet.loss import *

class Layer():
    def __init__(self):
        pass

    def forward(self, X):
        '''
        如果可以的话，gradient尽量从forward中update
        但遇到activation内置的情况，可以在update_gradient中嵌入。
        :param X:
        :return:
        '''
        pass

    def update_gradient(self, grad):
        '''
        用来跟新gradient，（用于back prop的时候）
        :param grad:
        :return:
        '''
        pass

    @property
    def params(self):
        return 0

    @property
    def gradient(self):
        return 0

class Dense(Layer):
    def __init__(self, hidden_unit, input_shape, activation):

        '''
        :param hidden_unit:   # of hidden units
        :param input_shape:   input shape
        :param activation:    Activation Class
        '''

        super().__init__()
        # Normal initialization
        self.n, self.p = input_shape  # input_tensor shape
        self.m = hidden_unit

        self.w = np.random.normal(0, 1, (self.p, self.m))
        self.b = np.random.normal(0, 1, (self.n, self.m))

        self.activation = activation

    def forward(self, X):
        # Forward Propagation
        self.X = X
        output = np.matmul(self.X, self.w) + self.b
        output = self.activation.forward(output)

        return output

    def update_gradient(self, grad):
        '''
        :param grad: 链式法则传过来的上一层的gradient
        :return:
        '''

        # print("Input gradient", grad.shape)
        self.input_gradient_after_activation = np.multiply(grad, self.activation.gradient)
        self.grad_w = np.matmul(self.X.T, self.input_gradient_after_activation)
        self.grad_X = np.matmul(self.input_gradient_after_activation, self.w.T)
        self.grad_b = self.input_gradient_after_activation

        assert self.grad_w.shape == self.w.shape
        assert self.grad_b.shape == self.b.shape
        assert self.grad_X.shape == self.X.shape

        self.__model_gradient = self.grad_X

    @property
    def params(self):
        return {'w': self.w,
                'b': self.b}

    @property
    def gradient(self):
        '''

        :return: X -- model gradient for backprop
        '''
        return {"w": self.grad_w,
                "b": self.grad_b,
                "X": self.grad_X}


# class Input(Layer):
#     def __init__(self, input_shape):
#         super().__init__()
#         self.n, self.p = input_shape  # input_tensor shape
#
#     def forward(self, input_tensor):
#         self.input_tensor = input_tensor
#         return input_tensor


class Conv2D(Layer):
    def __init__(self, input_shape, filters, kernel_size, strides,
                 padding, activation):
        '''
        :param input_shape:     [channel, row, col]
        :param filters:         kernel 的维度，记为 p，同时也是输出的维度
        :param kernel_size:     kernel 的大小，记为 (u, v)
        :param strides:         (a, b) (向右stride，向下stride) a,b >= 1
        :param padding:         "valid" means no padding;
                                "causal" means;
                                "same" means output has the same length as input
                                ** TODO 先写valid 无padding，之后加上padding **
        :param activation:      activation function
        '''

        super().__init__()
        self.c, self.x, self.y = input_shape
        self.f = filters
        self.u, self.v = kernel_size
        self.sx, self.sy = strides
        self.px = self.py = 0 # TODO 有了padding后，改为padding
        self.activation = activation

        # Initialize the kernel
        ## 我们首先计算，CNN 后的新的图片大小
        self.x_new = int((self.x - self.f + 2 * self.px) / self.sx + 1)
        self.y_new = int((self.y - self.f + 2 * self.py) / self.sy + 1)
        ## w 是直接乘到原来的矩阵上面的，它的大小由 kernel 决定
        self.w = np.random.normal(0, 1, (self.f, self.u, self.v))
        ## b 是加在已经取过 sum 了的上面，它的大小由CNN 后的大小决定
        print(self.x_new)
        print(self.y_new)
        self.b = np.random.normal(0, 1, (self.f, self.x_new, self.y_new))


    # 由于我们有batch，所以还需要添加一些大的matrix
    def W(self, n):
        '''
        注意了，这个大的W 只是为了计算n batch的时候方便使用，
        update的时候，我们只update 小的 self.w
        TODO
        :param n:
        :return:
        '''
        return np.repeat(self.w, n)

    def forward(self, X):
        '''
        :param X: X is a 4d tensor, [batch, channel, row, col]
        # TODO 之后完善有channel摆放位置不同的情况。
        :return:
        '''

        # Check 一下所有的维度是否正确
        size = X.shape
        self.n = size[0]
        assert self.c == size[1]
        assert self.x == size[2]
        assert self.y == size[3]

        # 首先 生成新的那个矩阵，同样是 [batch, f, x_new, y_new] 4D
        output = np.zeros((self.n, self.c, self.x_new, self.y_new))

        # TODO 如何同时做batch个？
        for image_idx, image in enumerate(X):
            for i in range(self.x_new):
                for j in range(self.y_new):
                    x_start = int(i * self.sx)
                    x_end = int(x_start + self.u)
                    y_start = int(j * self.sy)
                    y_end = int(y_start + self.v)

                    # Computation
                    clip = image[:, x_start: x_end, y_start: y_end]
                    weighted_clip = np.multiply(clip, self.w)
                    sum_of_weighted_clip = np.sum(weighted_clip)
                    output_scalar = sum_of_weighted_clip + self.b[:,i,j]

                    output[image_idx, :, i, j] = output_scalar

        print(output)
        return output

    def update_gradient(self, grad):
        # TODO write CNN Gradients
        self.grad_w = np.ones(self.w.shape)
        self.grad_b = np.ones(self.b.shape)
        self.grad_x = np.ones(self.x.shape)
        pass

    @property
    def gradient(self):
        return {"w": self.grad_w,
                "b": self.grad_b,
                "x": self.grad_x}

    @property
    def params(self):
        return {"w": self.w,
                "b": self.b}

class MaxPooling2D(Layer):
    def __init__(self, input_shape, pool_size, strides, padding):
        '''
        :param function: could be np.max, np.min, np.average, etc.
        '''
        super().__init__()
        self.c, self.x, self.y = input_shape    #
        self.u, self.v = pool_size    #
        self.sx, self.sy = strides              #
        self.px = self.py = 0                   # TODO 有了padding后，改为padding

        self.x_new = int((self.x - self.c + 2 * self.px) / self.sx + 1)
        self.y_new = int((self.y - self.c + 2 * self.py) / self.sy + 1)

    def forward(self, X):

        # Check 一下所有的维度是否正确
        size = X.shape
        self.n = size[0]
        assert self.c == size[1]
        assert self.x == size[2]
        assert self.y == size[3]

        # 首先 生成新的那个矩阵，同样是 [batch, c, x_new, y_new] 4D
        output = np.zeros((self.n, self.c, self.x_new, self.y_new))

        # Mapping 的作用是记录最大值所在的位置
        # 当我们在做back prop的时候，就直接使用这个mapping，找到原来的位置，这样我们就可以直接把gradient 填充到原来的大矩阵里了
        # mapping的大小，是pool后的小矩阵的大小，不包括n。最后的2用来存储坐标。
        # gradient 的大小，是原来矩阵的大小。
        self.mapping = np.zeros((self.c, self.x_new, self.y_new, 2))
        self.__gradient = np.zeros((self.n, self.c, self.x, self.y))

        for image_idx, image in enumerate(X):
            for channel_idx in range(self.c):
                for i in range(self.x_new):
                    for j in range(self.y_new):

                        x_start = int(i * self.sx)
                        x_end = int(x_start + self.u)
                        y_start = int(j * self.sy)
                        y_end = int(y_start + self.v)

                        # Forward-prop
                        clip = image[channel_idx, x_start: x_end, y_start: y_end]
                        output[image_idx, channel_idx, i, j] = np.max(clip)

                        # Backward-prop
                        maximum_x = int(np.argmax(clip)/clip.shape[0]) + x_start
                        maximum_y = np.argmax(clip) % clip.shape[0] + y_start
                        self.mapping[channel_idx, maximum_x, maximum_y] = 1  # TODO 问题！！ 这里回去的时候应该取哪一个呢？如果是batch=n的话
        return output

    def update_gradient(self, grad):
        '''
        上面传过来的 gradient 是小性状的，我们需要把小gradient 重新mapping回大的gradient！！！
        :param grad:
        :return:
        '''
        #
        # for image_idx, image in enumerate(X):
        #     for channel_idx in range(self.c):
        #         for i in range(self.x_new):
        #             for j in range(self.y_new):
        #                 x_start = int(i * self.sx)
        #                 x_end = int(x_start + self.u)
        #                 y_start = int(j * self.sy)
        #                 y_end = int(y_start + self.v)
        #
        #                 # Forward-prop
        #                 clip = image[channel_idx, x_start: x_end, y_start: y_end]
        #                 output[image_idx, channel_idx, i, j] = np.max(clip)
        #
        #                 # Backward-prop
        #                 maximum_x = int(np.argmax(clip) / clip.shape[0]) + x_start
        #                 maximum_y = np.argmax(clip) % clip.shape[0] + y_start
        #                 self.mapping[image_idx, channel_idx, maximum_x, maximum_y] = 1
        #
        pass