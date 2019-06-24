"""
Can further change how optimizers work.

Further generalize the model into capability of coping with different w's and b's
"""
import numpy as np
from otter.dam.structure import Variable

__all__ = ["GradientDescent", "StochasticGradientDescent",
           "RMSProp", "Adam"]


class Optimizer:
    def __init__(self):
        pass

    def update_once(self, x: Variable):
        pass

    def gradient_parser(self, grad, method, minibatch=-1):

        """
        :param grad:
        :param method:      string  full, stochastic, minibatch
        :param minibatch:   int     minibatch
        :return:            []      average gradient
        """

        n = grad.shape[0]
        if method == 'full':
            # axis = 1, col-wise average, for param share b
            # axis = 0, row-wise average, for normal gradients
            n_grad = np.average(grad, axis=0)

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

        # Gradient Clipping
        x.value = x.value - self.learning_rate * gradient


class StochasticGradientDescent(Optimizer):
    """
    In stochastic Gradient Descent,
    we actually need to use gradient parsers for both param_share and not
    share cases.
    """

    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def update_once(self, x: Variable):
        gradient = self.gradient_parser(x.gradient, 'stochastic')
        x.value = x.value - self.learning_rate * gradient


class RMSProp(Optimizer):
    """
    The RMSProp method
    """

    def __init__(self, learning_rate, decay_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.r = 0

    def update_once(self, x: Variable):

        if x.first_optimize:
            x.r = 0
            x.first_optimize = False

        gradient = self.gradient_parser(x.gradient, 'full')
        x.r = self.decay_rate * x.r + (1 - self.decay_rate) * np.multiply(gradient, gradient)
        gradient = - np.multiply(self.learning_rate / np.sqrt(1e-6 + x.r), gradient)
        x.value = x.value - gradient


class Adam(Optimizer):
    """
    Adam Optimizer
    """

    def __init__(self, learning_rate, decay_1=0.9, decay_2=0.999):
        super().__init__()
        self.learning_rate = learning_rate
        self.decay_1 = decay_1
        self.decay_2 = decay_2

    def update_once(self, x: Variable):

        if x.first_optimize:
            x.adam_r = 0.
            x.adam_t = 0
            x.adam_s = 0.

            x.first_optimize = False

        gradient = self.gradient_parser(x.gradient, 'full')

        x.adam_t += 1
        x.adam_s = self.decay_1 * x.adam_s + (1 - self.decay_1) * gradient
        x.adam_r = self.decay_2 * x.adam_r + (1 - self.decay_2) * np.multiply(gradient, gradient)
        s_hat = x.adam_s / (1 - self.decay_1 ** x.adam_t)
        r_hat = x.adam_r / (1 - self.decay_2 ** x.adam_t)

        gradient = - self.learning_rate * s_hat / (np.sqrt(r_hat) + 1e-6)
        x.value = x.value - gradient

