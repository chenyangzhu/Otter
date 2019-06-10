"""
Can further change how optimizers work.

Further generalize the model into capability of coping with different w's and b's
"""
import numpy as np
from otter.dam.structure import Variable


class Optimizer:
    def __init__(self):
        pass

    def update_once(self, x: Variable):
        pass


class GradientDescent(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def update_once(self, x: Variable):

        if x.param_share:
            gradient = np.average(x.gradient, axis=1).reshape((x.gradient.shape[0], 1))
        else:
            gradient = x.gradient

        x.value -= self.learning_rate * gradient