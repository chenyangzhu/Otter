import numpy as np
from otter.dam.structure import Variable
from otter.optimizer import Optimizer

class Graph:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def update_gradient(self, x: Variable):

        '''
        update gradient 是 update all children's gradients.
        grad 是从上面传入的gradient，如果为空，则自动变为np.ones
        这个时候默认这个Variable是parent节点
        :return:
        '''

        # IMPORTANT
        # 我们每次iteration 都是在更新 lchild and rchild 的 gradient！！！
        # 自己的gradient 永远是自己的parent那边更新的，
        # 如果没有parent，就直接是用初始化的时候的gradient！！！
        if x.lchild != None:

            x.update_lchild_gradient()
            self.update_gradient(x=x.lchild)

        if x.rchild != None:
            x.update_rchild_gradient()
            self.update_gradient(x=x.rchild)

    def update_gradient_with_optimizer(self, x: Variable, optimizer: Optimizer):

        if x.lchild != None:

            x.update_lchild_gradient()
            self.update_gradient_with_optimizer(x=x.lchild, optimizer=optimizer)

        if x.rchild != None:
            x.update_rchild_gradient()
            self.update_gradient_with_optimizer(x=x.rchild, optimizer=optimizer)

        # After updating the children's gradient
        # We update our own gradients if trainable

        if x.trainable:
            optimizer.update_once(x)