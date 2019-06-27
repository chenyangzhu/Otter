from otter import Variable
import numpy as np
import gzip
from otter.layers.convolution import Conv2D
from otter.ops.activation import relu
import matplotlib.pyplot as plt
from otter.dam.graph import Graph

w = [[[[-0.0394, -0.1065,  0.0439, -0.0088, -0.0170],
          [ 0.1969, -0.0443,  0.0493,  0.0645,  0.1785],
          [-0.1905,  0.1350,  0.0057, -0.1409,  0.1120],
          [-0.0441, -0.1194,  0.1239, -0.0306, -0.0034],
          [ 0.0288,  0.1108, -0.1808, -0.0666, -0.1213]]]]


def read_data():
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    paths = []
    for i in range(len(files)):
        paths.append('/home/klaus/Desktop/Otter/dataset/fashion-mnist/' + files[i])

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = read_data()

conv = Conv2D(1, (5, 5), relu, (2, 2), bias=False)
conv.w = w

t = Variable(x_train[0].reshape(1, 1, 28, 28))
ans = conv.forward(t)
print(conv.w)
print(ans)
print(ans.shape)
plt.imshow(ans.value[0][0])
plt.show()

g = Graph()

g.update_gradient(ans)

print(t.gradient)