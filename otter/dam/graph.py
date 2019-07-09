import numpy as np
from otter.dam.structure import Variable
from otter.optimizer import Optimizer
from otter._hyperparam import *
import otter.ops as ops


class Graph:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def back_propagate_with_optimizer(self, x: Variable, optimizer: Optimizer):

        if x is not None:
            # print(type(x))

            # Gradient Clipping
            mask = (x.gradient.value < GRADIENT_CLIPPING_THRESHOLD).astype(int)
            mask = np.multiply(mask, (x.gradient.value > -GRADIENT_CLIPPING_THRESHOLD).astype(int))
            mask = Variable(mask)
            contra_mask = Variable(np.array(1)) - mask
            x.gradient = ops.multiply(mask, x.gradient) + ops.dot(contra_mask, ops.constant(GRADIENT_CLIPPING_THRESHOLD))

            if x.back_prop is not None:
                # which means x is an input node
                x.back_prop(x)

            if x.trainable:
                optimizer.update_once(x)

            self.back_propagate_with_optimizer(x.lchild, optimizer)
            self.back_propagate_with_optimizer(x.rchild, optimizer)

    def update_gradient(self, x: Variable):
        if x.back_prop is not None:
            x.back_prop(x)

        if x.lchild is not None:
            self.update_gradient(x.lchild)

        if x.rchild is not None:
            self.update_gradient(x.rchild)

    def set_and_update_gradient(self, x: Variable, gradient):
        assert x.gradient.shape == gradient.shape
        x.gradient = gradient
        self.update_gradient(x)
