from otter.layers import common
import numpy as np
from otter.dam.structure import Variable


# CNN
class Conv2D(common.Layer):
    def __init__(self, in_channel, out_channel, kernel_size, stride=(1, 1),
                 padding=(0, 0), bias=True, trainable=True):
        """
        Convolution Layer 2D
        :param in_channel:      Int:    Number of input channels
        :param out_channel:     Int:    Number of output channels
        :param kernel_size:     Tuple:  kernel_size
        :param stride:          Tuple:  stride, default (1, 1)
        :param padding:         Tuple:  padding, default (0, 0)
        :param bias:            Bool:
        :param trainable:       Bool:
        """

        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding                  # TODO Add padding
        self.bias = bias                        # whether or not to add bias
        self.trainable = trainable

        # Initialize the kernel
        self.w = Variable(np.random.normal(0, 1, (self.out_channel, self.in_channel,
                                                  self.kernel_size[0], self.kernel_size[1])),
                          trainable=trainable)

        # To be compatible with the previous setup,
        # the shape of b needs to have a 1 on the last dimension.
        # Therefore, we need here the reversed, and on later implementations,
        # We need to add a transpose to the addition.


    def train_forward(self, X: Variable):
        '''
        :param X: X is a 4d tensor, [batch, channel, row, col]
        # TODO add channel in different places
        :return:
        '''
        self.n, _, self.x, self.y = X.shape
        assert self.in_channel == X.shape[1]

        # We first calculate the new matrix size.
        self.x_new = int((self.x - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)
        self.y_new = int((self.y - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1] + 1)

        # Check 一下所有的维度是否正确

        output = X.conv2d(self.w, stride=self.stride, padding=self.padding)

        # Add bias if necessary
        if self.bias:
            self.b = Variable(np.random.normal(0, 1, list(reversed((1, self.out_channel,
                                                                    self.x_new, self.y_new)))),
                              trainable=self.trainable, param_share=True)
            output1 = output + self.b
            return output1

        return output

    def predict_forward(self, x):
        return self.train_forward(x)


class MaxPooling2D(common.Layer):
    def __init__(self, input_shape, pool_size, strides, padding):
        """
        :param input_shape:
        :param pool_size:
        :param strides:
        :param padding:
        """
        super().__init__()
        self.c, self.x, self.y = input_shape    #
        self.u, self.v = pool_size              #
        self.sx, self.sy = strides              #
        self.px = self.py = 0                   # TODO 有了padding后，改为padding

        self.x_new = int((self.x - self.c + 2 * self.px) / self.sx + 1)
        self.y_new = int((self.y - self.c + 2 * self.py) / self.sy + 1)


        self.trainable = False

    def train_forward(self, X):

        # Check 一下所有的维度是否正确
        size = X.shape
        self.n = size[0]
        assert self.c == size[1]
        assert self.x == size[2]
        assert self.y == size[3]

        # 首先 生成新的那个矩阵，同样是 [batch, c, x_new, y_new] 4D
        output = Variable(np.zeros((self.n, self.c, self.x_new, self.y_new)),
                          lchild=X)

        # Mapping 的作用是记录最大值所在的位置
        # 当我们在做back prop的时候，就直接使用这个mapping，找到原来的位置，这样我们就可以直接把gradient 填充到原来的大矩阵里了
        # mapping的大小，是pool后的小矩阵的大小，不包括n。最后的2用来存储坐标。
        # gradient 的大小，是原来矩阵的大小。

        output.mapping = np.zeros((self.n, self.c, self.x_new, self.y_new, 2))

        output.size = [self.n, self.c, self.x_new, self.y_new]

        for image_idx, image in enumerate(X.value):
            for channel_idx in range(self.c):
                for i in range(self.x_new):
                    for j in range(self.y_new):

                        x_start = int(i * self.sx)
                        x_end = int(x_start + self.u)
                        y_start = int(j * self.sy)
                        y_end = int(y_start + self.v)

                        # Forward-prop
                        clip = image[channel_idx, x_start: x_end, y_start: y_end]
                        output.value[image_idx, channel_idx, i, j] = np.max(clip)

                        # Backward-prop
                        maximum_x = int(np.argmax(clip)/clip.shape[0]) + x_start
                        maximum_y = np.argmax(clip) % clip.shape[0] + y_start

                        # 把最大值的位置的坐标记录在mapping里
                        output.mapping[image_idx, channel_idx, i, j, 0] = maximum_x
                        output.mapping[image_idx, channel_idx, i, j, 1] = maximum_y

        output.back_prop = output.back_maxpooling2d()

        return output

    @property
    def output_shape(self):
        return self.c, self.x_new, self.y_new


class Flatten(common.Layer):
    def __init__(self):
        super().__init__()

    def train_forward(self, X):
        self.n, self.c, self.x, self.y = X.shape
        output = Variable(X.value.reshape((self.n, self.c * self.x * self.y)),
                        lchild=X)
        output.back_prop = output.back_flatten
        output.flatten_grad_parser = {"shape": X.shape}
        return output

    # @property
    # def output_shape(self):
    #     return self.c*self.x*self.y