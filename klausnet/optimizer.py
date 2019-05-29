class Optimizer:
    def __init__(self):
        pass

    def optimize(self, Layer):
        pass


class GradientDescent(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def update_once(self, layer):
        layer.w = layer.w - self.learning_rate * layer.gradient['w']
        layer.b = layer.b - self.learning_rate * layer.gradient['b']