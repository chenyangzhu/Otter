from otter.dam.structure import *
from otter.dam.graph import *
from otter.activation import sigmoid

with Graph() as graph:

    print(type(graph))

    a = Variable(np.array([[10, 15], [4, 8], [1, 2]]))
    b = Variable(np.array([[1, 2], [4, 8], [2, 4]]))
    c = Variable(np.array([[4, 5], [2, 13], [1, 8]]))
    d = Variable(np.array([[4, 3], [6, 10], [2, 10]]))

    # Test sigmoid activation
    e = a.sub(b)
    print("e", e.value)

    j = sigmoid(e)
    print("j", j.value)

    f = c.dot(d.T())

    g = f.dot(j)

    h = Variable(np.array([[1, 2],[3, 4],[5, 6]]))

    i = g.multiply(h)

    print("i", i.value)

    k = i.sum(axis=1)
    print("k", k.value)
    graph.update_gradient(k)

    print("a", a.gradient)
    print("b", b.gradient)
    print("c", c.gradient)
    print("d", d.gradient)
    print("e", e.gradient)
    print("f", f.gradient)
    print("g", g.gradient)
    print("h", h.gradient)
    print("i", i.gradient)
    print("j", j.gradient)

