import numpy as np

class Variable:

    def __init__(self, x, lchild=None, rchild=None, path='input'):
        '''

        :param x:
        :param lchild:
        :param rchild:
        :param path:    用来存储这个变量得到的方式，用来计算 back-prop
        '''

        self.x = x
        self.lchild = lchild
        self.rchild = rchild
        self.parent = None
        self.gradient = np.ones(self.x.shape)
        self.path = path  # 记录的是自己是如何得到的，与之后的 parent 无关

        # In-grad 的形状不定，应该与之后的parent一致
        self.in_grad = None
        # out-grad 的形状确定，必须和本层的input的shape一样
        self.out_grad = np.ones(self.x.shape)

    @property
    def value(self):
        return self.x

    @property
    def shape(self):
        return self.x.shape

    '''
    下面的这些操作，是对自身的变换，但仍然需要增加一个parent节点，把自己注册为 lchild
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
        return self.parent

    # def minimum(self):

    '''
    下面的这些操作，全部都是为二叉树加一个parent节点，
    '''

    def add(self, y):
        """
        :param y: Also a variable
        :return:
        """
        self.parent = Variable(self.value + y.value,
                               lchild=self, rchild=y,
                               path='add')
        self.parser = {}
        return self.parent

    def sub(self, y):
        """
        :param y: Also a variable
        :return:
        """
        self.parent = Variable(self.value - y.value,
                               lchild=self, rchild=y,
                               path='sub')
        self.parser = {}
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

    def update_lchild_gradient(self):
        '''
        每次backprop的时候，我们都是直接更新子节点的。
        :return:
        '''

        if self.path == "add":
            self.lchild.gradient = self.gradient
        elif self.path == "sub":
            self.lchild.gradient = self.gradient
        elif self.path == "T":
            self.lchild.gradient = self.gradient.T
        elif self.path == "dot":
            self.lchild.gradient = np.matmul(self.gradient,
                                             self.dot_grad_parser['y'].T)
        elif self.path == "multiply":
            self.lchild.gradient = np.multiply(self.gradient,
                                               self.multiply_grad_parser['y'])


    def update_rchild_gradient(self):

        '''
        每次 back-prop 的时候，我们都是直接更新子节点的。
        :return:
        '''

        if self.path == "add":
            self.rchild.gradient = self.gradient
        elif self.path == "sub":
            self.rchild.gradient = - self.gradient
        elif self.path == "T":
            self.rchild.gradient = self.gradient.T
        elif self.path == "dot":
            self.rchild.gradient = np.matmul(self.dot_grad_parser['x'].T,
                                             self.gradient)
        elif self.path == "multiply":
            self.rchild.gradient = np.multiply(self.gradient,
                                               self.multiply_grad_parser['x'])