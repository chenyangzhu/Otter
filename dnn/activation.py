import numpy as np

class Activation:
    def __init__(self):
        pass

    def forward(self, X):
        return X

    @property
    def gradient(self):
        return 0


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.X = X
        self.S = 1 / (1 + np.exp(-X))
        return self.S

    @property
    def gradient(self):
        return self.S * (1 - self.S)