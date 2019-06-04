from klausnet.layers import base
import numpy as np

# CNN
class Conv2D(base.Layer):
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
        self.learnable = learnable

        # Initialize the kernel
        # 这个Layer的输出是一个 [n, c_new, x_new, y_new] 的matrix

        # 我们首先计算，CNN 后的新的图片大小
        self.x_new = int((self.x - self.f + 2 * self.px) / self.sx + 1)
        self.y_new = int((self.y - self.f + 2 * self.py) / self.sy + 1)

        # w 是直接乘到原来的矩阵上面的，它的大小由 kernel 决定
        # self.c 是为了和输入的矩阵的维度匹配
        # self.f 就是我一共用了多少个filter，所以放在最外层。
        self.w = np.random.normal(0, 1, (self.f, self.c, self.u, self.v))

        # b 是加在已经取过 sum 了的上面，它的大小由 CNN 后的大小决定
        self.b = np.random.normal(0, 1, (self.f, self.x_new, self.y_new))

    def train_forward(self, X):
        '''
        :param X: X is a 4d tensor, [batch, channel, row, col]
        # TODO 之后完善有channel摆放位置不同的情况。
        :return:
        '''

        # Check 一下所有的维度是否正确
        size = X.shape
        self.X = X                      # 大 X 是数据，小x是维度
        self.n = size[0]
        assert self.c == size[1]
        assert self.x == size[2]
        assert self.y == size[3]

        # 首先 生成新的矩阵，同样是 [batch, f, x_new, y_new] 4D
        output = np.zeros((self.n, self.f, self.x_new, self.y_new))

        # mapping 的维度和输出的维度一致,最后一维是两个
        self.x2w = dict()
        self.w2x = dict()
        self.mapping_new2old = np.zeros((self.n, self.f, self.x_new, self.y_new, 2))
        self.mapping_old2new = np.zeros((self.n, self.f, self.x, self.y, 2))


        # grad_x 的维度和进入矩阵的维度一致
        self.grad_x = np.zeros(X.shape)         # [n, c, x, y]
        self.grad_w = np.zeros(self.w.shape)    # [f, c, u, v]
        self.grad_b = np.zeros(self.b.shape)    # [f, x_new, y_new]

        # TODO 如何同时做batch个？
        record = True
        for image_idx, image in enumerate(X):
            for filter_idx in range(self.f):

                for i in range(self.x_new):
                    for j in range(self.y_new):

                        x_start = int(i * self.sx)
                        x_end = int(x_start + self.u)
                        y_start = int(j * self.sy)
                        y_end = int(y_start + self.v)

                        if record:
                            for ix in range(self.u):
                                for jx in range(self.v):
                                    try:
                                        self.w2x[(0+ix, 0+jx)].append((x_start+ix, y_start+jx))
                                        self.x2w[(x_start+ix, y_start+jx)].append((0+ix, 0+jx))
                                    except:
                                        self.w2x[(0+ix, 0+jx)] = [(x_start+ix, y_start+jx)]
                                        self.x2w[(x_start+ix, y_start+jx)] = [(0+ix, 0+jx)]

                            # record是记录所有的mapping，只需要循环过一次后，其他都是一样的
                        # end record

                        # Computation
                        clip = image[:, x_start: x_end, y_start: y_end]
                        weighted_clip = np.multiply(clip, self.w[filter_idx])
                        sum_of_weighted_clip = np.sum(weighted_clip)
                        output_scalar = sum_of_weighted_clip + self.b[filter_idx, i, j]

                        # 记录Mapping
                        maximum_x = int(np.argmax(clip)/clip.shape[0]) + x_start
                        maximum_y = np.argmax(clip) % clip.shape[0] + y_start

                        # 把最大值的位置的坐标记录在mapping里
                        self.mapping_new2old[image_idx, filter_idx, i, j, 0] = maximum_x
                        self.mapping_new2old[image_idx, filter_idx, i, j, 1] = maximum_y

                        self.mapping_old2new[image_idx, filter_idx, maximum_x, maximum_y, 1] = i
                        self.mapping_old2new[image_idx, filter_idx, maximum_x, maximum_y, 1] = j

                        output[image_idx, filter_idx, i, j] = output_scalar

                    # end for j
                # end for i
                record = False
            # end for filter_idx
        # end for image_idx

        return output

    def update_gradient_w(self, grad):

        self.grad_w = np.ones(self.w.shape)  # f c u v

        for filter_idx in range(self.f):  # 对于每一个filter

            for channel_idx in range(self.c):  # 对于这个filter的每一层

                for i in range(self.u):
                    for j in range(self.v):  # 对于这一层filter的每一个元素

                        summation = 0
                        for each_idx in self.w2x[(i, j)]:  # 我们去找w2x当中与他对应的这些坐标

                            for image_idx in range(self.n):
                                # self.X  [n, c, x, y]
                                summation += self.X[image_idx, channel_idx][
                                                 self.mapping_new2old[each_idx[0], each_idx[1]]] * \
                                             grad[image_idx, filter_idx][each_idx]

                            # End for each image

                        self.grad_w[i, j] = summation / self.n  # 这里直接使用了average 的 gradient 处理方式

                        # End for 某个 w[i, j] 所对应的x

                # End for i,j 某一层(channel) filter 中的所有元素

            # End for 这个filter的每一层channel

        # End for 每一个filter

    def update_gradient_x(self, grad):

        self.grad_x = np.ones(self.X.shape)  # n c x y

        for image_idx in (self.n):  # 不像之前做grad_w 现在每个grad对于不同的x是不同的

            for channel_idx in range(self.c):  # 对于每一层channel

                for i in range(self.x):
                    for j in range(self.y):  # 对x中的每一个元素

                        summation = 0

                        for each_idx in self.x2w[(i, j)]:  # 去找到所有对应的w

                            for filter_idx in range(self.f):
                                # X 中每个元素的gradient 等于 sum of 对应的 w 乘上传来的grad
                                # 因为是linear的，所以先求 avg grad 和后求是一样的。
                                summation += self.w[channel_idx, filter_idx][each_idx] * grad[image_idx, filter_idx][self.mapping_old2new[i, j]]

                            # End for each filter

                        # End for each corresponding w

                        self.grad_x[image_idx, channel_idx, i, j] = summation

                # End for each element in one channel of x

            # End for each channel

        # End for each image

    def update_gradient(self, grad, method, minibatch=-1):
        '''

        :param grad: [n, f, x_new, y_new]
        :return:
        '''

        # TODO write CNN Gradients

        self.update_gradient_x(grad)
        self.update_gradient_w(grad)
        self.grad_b = self.average_gradient(grad, method, minibatch)

    @property
    def gradient(self):
        return {"w": self.grad_w,
                "b": self.grad_b,
                "x": self.grad_x,
                "back": self.grad_x}

    @property
    def params(self):
        return {"w": self.w,
                "b": self.b}


class MaxPooling2D(base.Layer):
    def __init__(self, input_shape, pool_size, strides, padding):
        '''

        :param input_shape:
        :param pool_size:
        :param strides:
        :param padding:
        '''
        super().__init__()
        self.c, self.x, self.y = input_shape    #
        self.u, self.v = pool_size              #
        self.sx, self.sy = strides              #
        self.px = self.py = 0                   # TODO 有了padding后，改为padding

        self.x_new = int((self.x - self.c + 2 * self.px) / self.sx + 1)
        self.y_new = int((self.y - self.c + 2 * self.py) / self.sy + 1)

        # Pooling 本来就不需要 learning 和 update
        self.learnable = False

    def train_forward(self, X):

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

    def update_gradient(self, grad, method, minibatch=-1):
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
                        # Tuple 的目的是为了把 [] 变为 () 从而可以调用
                        self.grad_x[image_idx, channel_idx][tuple(self.mapping[i, j])] = grad[image_idx, channel_idx, i, j]
    # end update gradient

    @property
    def gradient(self):
        '''
        Pooling Layer 只有一个往前的gradient，并不需要每次自己update。
        :return:
        '''
        return {"back": self.grad_x}


class Flatten(base.Layer):
    def __init__(self):
        super().__init__()
        # There's no need to specify the input shape
        self.learnable = False

    def train_forward(self, X):

        self.n, self.c, self.x, self.y = X.shape
        return X.reshape((self.n, self.c * self.x * self.y))

    def update_gradient(self, grad, method='full', minibatch=-1):
        '''
        :param grad:        进来的gradient是一个 (n, c*x*y) 维度的向量
        :param method:      这里的method必然是full
        :return:            [n, c, x, y] 的 gradient 矩阵全部map回去
        '''
        self.grad_x = grad.reshape((self.n, self.c, self.x, self.y))

    @property
    def gradient(self):
        return {'back': self.grad_x}