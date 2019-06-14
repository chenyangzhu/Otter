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

    def gradient_parser(self, grad, method, minibatch=-1):
        '''
        :param grad:
        :param method:      string  full, stochastic, minibatch
        :param minibatch:   int     minibatch的数量
        :return:            []      average gradient
        '''

        n = grad.shape[0]
        if method == 'full':
            # axis = 1, col-wise average
            # axis = 0, row-wise average

            n_grad = np.average(grad, axis=1).reshape((grad.shape[0], 1))

        elif method == 'stochastic':
            n_grad = grad[:, np.random.randint(0, n, 1)]

        elif method == 'minibatch':
            if minibatch <= 0 or minibatch > n:
                raise ValueError("Please specify a correct minibatch > 0.")
            else:
                n_grad = np.average(grad[np.random.randint(0, n, minibatch)], axis=1).reshape((grad.shape[0], 1))
        else:
            n_grad = 0
            raise ValueError("method could only be full, stochastic or minibatch.")

        return n_grad


class GradientDescent(Optimizer):
    def __init__(self, learning_rate, mini_batch=-1):
        super().__init__()
        self.learning_rate = learning_rate
        self.mini_batch = mini_batch

    def update_once(self, x: Variable):

        if x.param_share:
            gradient = self.gradient_parser(x.gradient, 'full', self.mini_batch)
        else:
            gradient = x.gradient

        x.value -= self.learning_rate * gradient


class StochasticGradientDescent(Optimizer):
    #TODO, rewrite the SGD, to keep track of only one gradient.
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def update_once(self, x: Variable):

        if x.param_share:
            gradient = self.gradient_parser(x.gradient, 'stochastic')
        else:
            gradient = x.gradient

        x.value -= self.learning_rate * gradient
