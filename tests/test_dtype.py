from otter.dam.structure import Variable
import numpy as np

a = Variable(np.ones(5, dtype=np.double),
             dtype=np.double)
b = Variable(np.ones(5, dtype=np.double),
             dtype=np.double)

c = a + b

print(c)