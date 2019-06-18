import numpy as np
from otter.dam.structure import Variable
from otter.optimizer import Optimizer

# GRADIENT_CLIPPING_THRESHOLD = 1e8
# GRADIENT_MAGNIFIER = 1e2


class Graph:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def update_gradient_with_optimizer(self, x: Variable, optimizer: Optimizer):
        # print(type(x))
        if x.back_prop is not None:

            # '''
            # Gradient Clipping
            # '''
            #
            # if np.sum(x.gradient ** 2) > GRADIENT_CLIPPING_THRESHOLD:
            #     x.gradient = np.sign(x.gradient) * GRADIENT_MAGNIFIER

            x.back_prop()

        if x.lchild is not None:
            self.update_gradient_with_optimizer(x.lchild, optimizer)

        if x.rchild is not None:
            self.update_gradient_with_optimizer(x.rchild, optimizer)

        # After updating the children's gradient
        # We update the value if trainable

        if x.trainable:
            optimizer.update_once(x)

    def update_gradient(self, x: Variable):
        if x.back_prop is not None:
            x.back_prop()

        if x.lchild is not None:
            self.update_gradient(x.lchild)

        if x.rchild is not None:
            self.update_gradient(x.rchild)
