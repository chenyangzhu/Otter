from otter.dam.structure import *
from otter.dam.math import *
from otter.dam.graph import *


a = Variable(np.array([10,15]))
b = Variable(np.array([1,2]))


print(type(a))
c = add(a, b)
print(type(c))
d = add(c, a)
print(d.gradient)
