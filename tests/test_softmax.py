
import numpy as np
from otter.ops.activation import softmax
from otter.dam.graph import Graph
# n=10
# m=2
#
# a = np.arange(n * m).reshape(n, m)
#
# eyed = a.reshape(n, m, 1) * np.eye(m)
#
# ones = a.reshape(n, m, 1) * np.ones(m)
# inverse_ones = a.reshape(n, 1, m) * np.ones((m, m))
#
# ds = eyed - np.multiply(ones, inverse_ones)
# avg_ds = np.average(ds, axis=0)
#
# print(ones)
# print(inverse_ones)
# print(eyed)
# print(ds)



from otter.dam.structure import Variable

a = Variable(np.random.normal(0, 1, (20, 10)))
#
b = softmax(a)
g = Graph()
# # c = a.sum(axis=1)
# # g.set_and_update_gradient(c, np.arange(20).reshape(20,1))
# g.update_gradient(b)
#
# print(a.gradient)
c = Variable(np.random.randint(0, 9, (20,1)))


sliced = b.slice(c.value.reshape((len(c.value),)), axis=1)

mid1=sliced.safe_log()
mid2=mid1.average()
mid3=mid2.neg()

g.update_gradient(mid3)

print(a.gradient)