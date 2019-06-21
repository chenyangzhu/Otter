import numpy as np
from otter import Variable
from otter.ops.loss import mean_squared_error
from otter.dam.graph import Graph
from otter.ops.activation import sigmoid, softmax

with Graph() as g:
    a = Variable(np.array([[10, 15], [4, 8], [1, 2]]))
    b = Variable(np.array([[1, 2], [4, 8], [2, 4]]))
    c = Variable(np.array([[4, 5], [2, 13], [1, 8]]))
    d = Variable(np.array([[4, 3], [6, 10], [2, 10]]))

    e = a.add(b)
    f = c.sub(d)

    h = sigmoid(e)
    i = softmax(f)

    ans = mean_squared_error(h, i)
    g.update_gradient(ans)

    print("ans", ans.gradient)
    print("f", f.gradient)
    print("e", e.gradient)
    print("d", d.gradient)
    print("c", c.gradient)
    print("i", i.gradient)
    print(i)

    print(ans)