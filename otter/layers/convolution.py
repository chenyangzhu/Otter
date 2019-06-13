from otter.layers import common
import numpy as np
from otter.dam.structure import Variable


# CNN
class Conv2D(common.Layer):
    def __init__(self, input_shape, filters, kernel_size, strides,
                 padding, activation, trainable=True):

        '''
        :param input_shape:     [channel, row, col]
        :param filters:         kernel 的维度，记为 p，同时也是输出的维度
        :param kernel_size:     kernel 的大小，记为 (u, v)
        :param strides:         (a, b) (向右stride，向下stride) a,b >= 1
        :param padding:         "valid" means no padding;
                                "causal" means;
                                "same" means output has the same length as input
                                TODO 先写valid 无padding，之后加上padding
        :param activation:      activation function
        :param learnable:       learnable=True，则在 gradient descent 的时候更新这层的param
                                如果False，则不更新
        '''

        super().__init__()
        self.c, self.x, self.y = input_shape
        self.f = filters        # Filter 数就是新的channel数。
        self.u, self.v = kernel_size
        self.sx, self.sy = strides
        self.px = self.py = 0  # TODO 有了padding后，改为padding
        self.activation = activation
        self.trainable = trainable

        # Initialize the kernel
        # 这个Layer的输出是一个 [n, c_new, x_new, y_new] 的matrix

        # 我们首先计算，CNN 后的新的图片大小
        self.x_new = int((self.x - self.f + 2 * self.px) / self.sx + 1)
        self.y_new = int((self.y - self.f + 2 * self.py) / self.sy + 1)

        # w 是直接乘到原来的矩阵上面的，它的大小由 kernel 决定
        # self.c 是为了和输入的矩阵的维度匹配
        # self.f 就是我一共用了多少个filter，所以放在最外层。
        self.w = Variable(np.random.normal(0, 1, (self.f, self.c, self.u, self.v)),
                          trainable=trainable)

        # To be compatible with the previous setup,
        # the shape of b needs to have a 1 on the last dimension.
        # Therefore, we need here the reversed, and on later implementations,
        # We need to add a transpose to the addition.
        self.b = Variable(np.random.normal(0, 1, list(reversed((1, self.f, self.x_new, self.y_new)))),
                          trainable=trainable, param_share=True)

    def train_forward(self, X: Variable):
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

        output = X.conv2d(self.w,
                          stride=(self.sx, self.sy),
                          padding=(self.px, self.py))
        output = self.activation(output)
        return output

    def predict_forward(self, X):
        return self.train_forward(X)

    @property
    def output_shape(self):
        return self.f, self.x_new, self.y_new


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