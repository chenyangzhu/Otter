import numpy as np

class Variable:

    def __init__(self, tensor, from1=None, from2=None, path='input'):
        '''

        :param tensor:
        :param from1:
        :param from2:
        :param path:    用来存储这个变量得到的方式，用来计算backprop
        '''
        self.x = tensor
        self.from1 = from1
        self.from2 = from2
        self.next = None

        self.gradient = np.ones(self.x.shape)
        self.path = path

    @property
    def value(self):
        return self.x

    @property
    def shape(self):
        return self.x.shape

    @property
    def T(self):
        return Variable(self.value.T, from1=None)

    def update_gradient(self):
        # Update gradients

        if self.next != None:

            if self.path == "add":
                self.from1.gradient = self.next.gradient
                self.from2.gradient = self.next.gradient

            elif self.path == "sub":
                self.from1.gradient = self.next.gradient
                self.from2.gradient = - self.next.gradient
        else:
            self.from1.gradient = np.ones(self.from1.shape)
            self.from2.gradient = np.ones(self.from2.shape)

    def add(self, y):
        """
        :param y: Also a variable
        :return:
        """
        self.next = Variable(self.value + y.value, from1=self, from2=y)
        self.next.path = 'add'
        return self.next

    def sub(self, y):
        """
        :param y: Also a variable
        :return:
        """
        self.next = Variable(self.value - y.value, from1=self, from2=y)
        self.next.path = 'sub'
        return self.next

    def dot(self, y):
        """
        @ - matmul
        :param y:
        :return:
        """
        return Variable(np.matmul(self.value, y.value),
                        from1=self, from2=y)

    def multiply(self, y):
        '''
        Element-wise multiplication
        :param y:
        :return:
        '''
        return Variable(np.multiply(self.value, y.value),
                        from1=self, from2=y)


if __name__ == "__main__":

    a = Variable(np.array(range(10)).reshape((5, 2)))
    b = Variable(np.array(range(10)).reshape((5, 2)))
    c = a.sub(b)

    c.update_gradient()
    print(c.gradient)
    print(a.gradient)
    print(b.gradient)
