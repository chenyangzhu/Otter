from otter.model import Model
from otter.layers.common import *
from otter.ops.activation import *
from otter.dam.structure import Variable


class One(Model):

    def __init__(self):
        super().__init__()
        self.layer1 = Dense(100, sigmoid)
        self.layer2 = Dense(10, softmax)

    def forward(self, x):
        x = self.layer1.train_forward(x)
        x = self.layer2.train_forward(x)
        return x
    #
    # def save(self, path):
    #     variables = self.__dict__
    #     for each in variables.keys():
    #         # If layers[each] is a Layer object, we call and save it
    #         if issubclass(variables[each].__class__, Layer):
    #             print("Saved")
    #             # variables[each].save_layer(path)
    #             # As long as it's a layer object, we store it.


if __name__ == "__main__":
    model = One()
    model.save()
    model.load()
    # __dir__())
    # print(a.__dict__)  # This function stores all the variables
    # layers = a.__dict__
    #
    # print(layers.keys())
    #
    # for each in layers.keys():
    #     print(layers[each])
    #     print(issubclass(layers[each].__class__, Layer))
    #     print(layers[each].__dict__)