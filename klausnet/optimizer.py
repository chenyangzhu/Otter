"""
Can further change how optimizers work.

Further generalize the model into capability of coping with different w's and b's
"""

class Optimizer:
    def __init__(self):
        pass

    def optimize(self, Layer):
        pass


class GradientDescent(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def update_once(self, each_layer):
        each_layer.w = each_layer.w - self.learning_rate * each_layer.gradient['w']
        each_layer.b = each_layer.b - self.learning_rate * each_layer.gradient['b']

