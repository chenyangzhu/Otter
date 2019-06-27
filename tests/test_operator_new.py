from otter import Variable
import numpy as np



def __add__(x: Variable, y: Variable):

    return x + y


a = Variable(np.ones(1))
b = Variable(np.ones(1))

print(a + b)