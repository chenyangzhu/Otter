import numpy as np


class Layer():
    def __init__(self):
        pass

    def forward(self, X):
        '''
        如果可以的话，gradient尽量从forward中update
        但遇到activation内置的情况，可以在update_gradient中嵌入。
        :param X:
        :return:
        '''
        pass

    def update_gradient(self, grad, method, minibatch=-1):
        '''
        用来跟新gradient，（用于back prop的时候）
        :param grad:
        :param method:      "full"
                            "stochastic"
        :param minibatch:  mini-batch 默认为 -1， 只有在stochastic的时候，才需要用minibatch
        :return:
        '''
        pass

    @property
    def params(self):
        return 0

    @property
    def gradient(self):
        return 0

    def average_gradient(self, grad, method, minibatch=-1):
        '''
        :param grad:
        :param method:      string  full, stochastic, minibatch
        :param minibatch:   int     minibatch的数量
        :return:            []      average gradient
        '''

        n = grad.shape[0]
        if method == 'full':
            n_grad = np.average(grad, axis=0).reshape((grad.shape[1],1))

        elif method == 'stochastic':
            n_grad = grad[np.random.randint(0, n, 1)].T

        elif method == 'minibatch':
            if minibatch <= 0 or minibatch > n:
                raise ValueError("Please specify a correct minibatch > 0.")
            else:
                n_grad = np.average(grad[np.random.randint(0, n, minibatch)])
        else:
            n_grad = 0
            raise ValueError("method could only be full, stochastic or minibatch.")

        return n_grad
