import otter as ot
import numpy as np


a = ot.Variable(np.array([[10, 15], [4, 8], [1, 2]]))
b = ot.Variable(np.array([[1, 2], [4, 8], [2, 4]]))
c = ot.Variable(np.array([[4, 5], [2, 13], [1, 8]]))
d = ot.Variable(np.array([[4, 3], [6, 10], [2, 10]]))

a + b
print(c + d)

e = ot.dot(c, d.T)
f = ot.multiply(c, d)
g = ot.Graph()
g.update_gradient(f)
print(c.gradient)