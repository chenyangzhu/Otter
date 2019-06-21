import numpy as np
from ..dam.structure import Variable
import json
import os

class Layer():
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def predict(self, x):
        return self.forward(x)

    def save_layer(self, path):
        """save layer data into a file (e.g., ./layer1.json
        the input path, must contains the target file that we want to write in.
        This file should be specified in the upper save_model function.

        Also notice that we will export *all* class params/variables of this instance.
        """
        result = self.__dict__
        delete_list = []
        # To create a delete_list is because during the following iteration,
        # it's no callable.

        # By iterating through all the results in the dictionary
        for each_key in result.keys():

            # Now, we need to turn all the Variable objects into a value list
            if isinstance(result[each_key], Variable):
                result[each_key] = result[each_key].value.tolist()

            # We need to delete all the functions
            if callable(result[each_key]):
                # If it's a function, delete it
                delete_list.append(each_key)

        # Now we delete what we don't want and marked before:
        for each in delete_list:
            result.pop(each)

        # Then we dump the results into a file
        with open(path, 'w') as outfile:
            json.dump(result, outfile)

    def read_layer(self, path):
        """Read layer data from a json file

        1. Read the json file and transform to string
        2. Parse json file using json package
        3. set attribute
        """
        with open(path) as json_file:
            data = json.load(json_file)
            for each_key in data.keys():
                # Set attribute
                self.__setattr__(each_key, data[each_key])



    @property
    def params(self):
        return 0


class Dense(Layer):
    def __init__(self, output_shape, activation, trainable=True):

        """
        :param output_shape:   # of hidden units
        :param activation:    an activation function
        """

        super().__init__()
        # Normal initialization
        self.m = output_shape
        self.trainable = trainable
        self.activation = activation
        self.initialize = True

    def forward(self, x: Variable):
        # Forward Propagation
        # print(x.shape)
        self.p = x.shape[1]  # input_tensor shape

        if self.initialize:
            self.w = Variable(np.random.normal(0, 1, (self.p, self.m)),
                              trainable=self.trainable, name='Dense_w')
            self.b = Variable(np.random.normal(0, 1, (self.m, 1)),
                              trainable=self.trainable, param_share=True,
                              name='Dense_b')
            self.initialize = False

        output = x.dot(self.w) + self.b.T()
        return self.activation(output)

    def predict(self, x: Variable):
        return self.forward(x)


class Dropout(Layer):
    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x: Variable):

        """
        The key of dropout, is reduce the dropout process to
        multiplying a random mask (0, 1).
        We can then simply use the standard grad-back-prop methods in the library.

        :param x:  for whichever dimension
        :return:
        """

        mask = np.ones(x.shape)
        dimension = len(x.shape)
        total_param = np.prod(x.shape)

        number_to_drop = np.int(total_param * self.dropout_rate)
        drop_coord = np.zeros((number_to_drop, dimension), dtype=np.int)

        for i in range(dimension):
            drop_coord[:, i] = np.random.randint(0, x.shape[i] - 1, number_to_drop).astype(int)

        for coord in drop_coord:

            mask[coord] = 0

        output = x.multiply(Variable(mask))

        return output

    def predict(self, x: Variable):

        return x


class BatchNormalization(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x: Variable):
        x_val = x.value
        self.mean = Variable(np.average(x.value))
        self.var_inv = Variable(np.array(np.var(x_val))).safe_inv()
        output = x.sub(self.mean).multiply(self.var_inv)
        return output

    def predict(self, x: Variable):

        output = x.sub(self.mean).multiply(self.var_inv)
        return output
