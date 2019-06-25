
import numpy as np


n=10
m=2

a = np.arange(n * m).reshape(n, m)

eyed = a.reshape(n, m, 1) * np.eye(m)

ones = a.reshape(n, m, 1) * np.ones(m)
inverse_ones = a.reshape(n, 1, m) * np.ones((m, m))

ds = eyed - np.multiply(ones, inverse_ones)
avg_ds = np.average(ds, axis=0)

print(ones)
print(inverse_ones)
print(eyed)
print(ds)