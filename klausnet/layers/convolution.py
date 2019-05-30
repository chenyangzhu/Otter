import klausnet.layers
import numpy as np

# CNN
class Conv2D(klausnet.layers.Layer):
    def __init__(self, input_shape, filters, kernel_size, strides,
                 padding, activation, learnable=True):
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
        :param learnable:       learnable=True，则在gradient descent 的时候更新这层的param，如果False，则不更新
        '''

        super().__init__()
        self.c, self.x, self.y = input_shape
        self.f = filters # TODO 问题，这个filter到底怎么理解，出来的是几维？？？
        self.u, self.v = kernel_size
        self.sx, self.sy = strides
        self.px = self.py = 0 # TODO 有了padding后，改为padding
        self.activation = activation
        self.learnable = learnable

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
                    output_scalar = sum_of_weighted_clip + self.b[:, i, j]

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
        self.u, self.v = pool_size              #
        self.sx, self.sy = strides              #
        self.px = self.py = 0                   # TODO 有了padding后，改为padding

        self.x_new = int((self.x - self.c + 2 * self.px) / self.sx + 1)
        self.y_new = int((self.y - self.c + 2 * self.py) / self.sy + 1)

        # Pooling 本来就不需要learning和update
        self.learnable = False

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
        self.mapping = np.zeros((self.n, self.c, self.x_new, self.y_new, 2))
        self.grad_x = np.zeros((self.n, self.c, self.x, self.y))

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

                        # 把最大值的位置的坐标记录在mapping里
                        self.mapping[image_idx, channel_idx, i, j, 0] = maximum_x
                        self.mapping[image_idx, channel_idx, i, j, 1] = maximum_y

        return output

    def update_gradient(self, grad):
        '''
        上面传过来的 gradient 是小性状的 [n, channel, x_new, y_new]
        ，我们需要把小gradient 重新mapping回大的gradient！！！
        小形状对应的就是mapping里的位置
        :param grad:
        :return:
        '''
        for image_idx in range(self.n):
            for channel_idx in range(self.c):
                for i in range(self.x_new):
                    for j in range(self.y_new):

                        # TODO 下面这个还没跑 不一定对
                        self.grad_x[image_idx, channel_idx][self.mapping[i, j]] = grad[image_idx, channel_idx, i, j]
    # end update gradient

    @property
    def gradient(self):
        '''
        Pooling Layer 只有一个往前的gradient，并不需要每次自己update。
        :return:
        '''
        return {"x": self.grad_x}


