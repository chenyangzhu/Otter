from otter.dam.structure import Variable
import numpy as np

class Operator:
    """
    Operator is always a class.
    All operators are connected one after another.
    """

    def __init__(self):
        pass

    @property
    def result(self):
        return Variable(0)

    @property
    def value(self):
        return self.result.value

    @property
    def gradient(self):
        return 0


class add(Operator):

    def __init__(self, from1, from2):
        super().__init__()
        self.from1 = from1
        self.from2 = from2

    @property
    def result(self):
        return Variable(self.from1.value + self.from2.value)

    @property
    def value(self):
        return self.result.value

    @property
    def gradient(self):
        return {'grad1': self.result.gradient,
                'grad2': self.result.gradient}


class matmul(Operator):

    def __init__(self, from1, from2):
        '''

        :param from1: Variable(n x p)
        :param from2: Variable(p x m)
        result : n x m
        '''
        super().__init__()

        assert from1.shape[1] == from2.shape[0]
        self.from1 = from1
        self.from2 = from2

    @property
    def result(self):
        return Variable(np.matmul(self.from1.value, self.from2.value))

    @property
    def value(self):
        return self.result.value

    @property
    def gradient(self):
        return {'grad1': np.matmul(self.result.gradient, self.from2.value.T),
                'grad2': np.matmul(self.from1.value.T, self.result.gradient)
                }