import numpy as np

class Variable:

    def __init__(self, tensor, from1=None, from2=None):
        self.x = tensor
        self.from1 = from1
        self.from2 = from2
        # self.__str__ = "otter_variable_" + name
        self.gradient = np.ones(self.x.shape)

    @property
    def value(self):
        return self.x

    @property
    def T(self):
        return Variable(self.value.T, from1=None)

    def update_gradient(self):
        # Update gradients
        self.gradient = self.new_variable.gradient
        y.gradient = self.new_variable.gradient

    def __add__(self, y):
        '''
        :param y: Also a variable
        :return:
        '''
        self.new_variable = Variable(self.value + y.value, from1=self, from2=y

        return self.new_variable

    def __sub__(self, y):
        '''
        :param y: Also a variable
        :return:
        '''
        return Variable(self.value - y.value, from1=self, from2=y)

    def __matmul__(self, y):
        '''
        @ - matmul
        :param y:
        :return:
        '''

        return Variable(np.matmul(self.value, y.value),
                        from1=self, from2=y)

    def __mul__(self, y):
        '''
        Element-wise multiplication
        :param y:
        :return:
        '''
        return Variable(np.multiply(self.value, y.value),
                        from1=self, from2=y)

    def concat(self, y, axis):
        return

if __name__ == "__main__":

    a = Variable(np.array(range(10)).reshape((2, 5)))
    b = Variable(np.array(range(10)).reshape((5, 2)))

    print((a @ b).value)
    print((a @ b).from2.value)

    print((a * b.T).value)

    