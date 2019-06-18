from otter.dam.structure import Variable
import numpy as np
from otter.dam.graph import Graph

a = Variable(np.arange(10).reshape(2, 5))

b = a.slice(np.array([0, 1, 0, 1, 0]), axis=0)

c = a.slice(np.array([0, 1]), axis=1)

g = Graph()

g.update_gradient(b)
print(a.gradient)