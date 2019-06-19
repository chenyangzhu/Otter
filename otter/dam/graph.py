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

        # Gradient Clipping
        GRADIENT_CLIPPING_THRESHOLD = 1e3
        mask = (x.gradient < GRADIENT_CLIPPING_THRESHOLD).astype(int)
        mask = np.multiply(mask, (x.gradient > -GRADIENT_CLIPPING_THRESHOLD).astype(int))
        contra_mask = 1 - mask
        x.gradient = np.multiply(mask, x.gradient) + contra_mask * GRADIENT_CLIPPING_THRESHOLD

        if x.back_prop is not None:
            # which means x is an input node
            x.back_prop()

        if x.trainable:
            optimizer.update_once(x)

        if x.lchild is not None:
            self.update_gradient_with_optimizer(x.lchild, optimizer)

        if x.rchild is not None:
            self.update_gradient_with_optimizer(x.rchild, optimizer)

        # After updating the children's gradient
        # We update the value if trainable

    def update_gradient(self, x: Variable):
        if x.back_prop is not None:
            x.back_prop()

        if x.lchild is not None:
            self.update_gradient(x.lchild)

        if x.rchild is not None:
            self.update_gradient(x.rchild)
