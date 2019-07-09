import numpy as np
from otter._hyperparam import *
import otter as ot


class Variable:

    def __init__(self, x, dtype=None, require_gradient=True, lchild=None, rchild=None,
                 trainable=False, param_share=False,
                 name=None):

        """
        :param x:
        :param lchild:
        :param rchild:
        :param path:  To record the path where this Variable is derived.
                      Notice that the paths here are by default 'input'
        :param trainable:
        :param param_share: Does this variable contains 0 that we need to average.
        """

        self.value = x

        # if dtype is None, we automatically generate dtype from x
        # Else we check if the declared datatype is the same as the input data x
        # if dtype is None:
        #     self.dtype = x.dtype
        # else:
        #     assert x.dtype == dtype

        self.lchild = lchild
        self.rchild = rchild
        if require_gradient:
            self.gradient = Variable(np.ones(self.value.shape), require_gradient=False)
        self.trainable = trainable
        self.param_share = param_share
        self.name = name
        self.back_prop = None

        # The variable is utilized to take notice of
        self.first_optimize = True

        '''
        The back_prop function is what we use to do back_propagation
        The function only exists when the Variable is derived from
        an input Variable.
        Thus, if there's no back_prop function, it must be the input
        and we no longer need to do the back_prop on it.
        '''

        self.back_prop = None

    def __str__(self):
        return "otter.Variable: " + str(self.name) + " " + " (" + self.value.__str__() + ")"

    '''
    The followings are two basic functions we provide to allow faster 
    computations and extra abilities.
    '''

    @property
    def shape(self):
        return self.value.shape

    def detach(self):
        """
        Sometimes, we don't really want this Variable to record any previous data,
        so we can use detach, to detach it from all previous nodes.
        """
        self.lchild = None
        self.rchild = None
        self.back_prop = None

    def copy(self):
        """
        The copy function allows us to create a copied version of this variable
        without interfering with the existing tensor graph

        By default, it's not trainable.
        """

        return Variable(self.value)

    def update_gradient(self, gradient):
        self.gradient = gradient.copy()

    '''
    These methods are only a transformation of the variable itself. They do not need any parent nodes.
    They are registered as lchild.
    '''

    @property
    def T(self):
        return ot.T(self)

    def __add__(self, y):
        return ot.ops.add(self, y)

    def __radd__(self, y):
        return ot.ops.add(self, y)

    def __sub__(self, y):
        return ot.ops.sub(self, y)

    def __rsub__(self, y):
        return ot.ops.sub(self, y)

    def __pow__(self, y):
        return ot.ops.pow(self, y)

    #
    # def back_maxpooling2d(self):
    #     n, c, x_new, y_new = self.size
    #
    #     grad_x = np.zeros((n, c, x_new, y_new))
    #
    #     for image_idx in range(n):
    #         for channel_idx in range(c):
    #             for i in range(x_new):
    #                 for j in range(y_new):
    #                     grad_x[image_idx, channel_idx][tuple(self.mapping[i, j])] = self.gradient[
    #                         image_idx, channel_idx, i, j]
    #     self.lchild.gradient = grad_x