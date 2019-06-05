from otter.dam.structure import *
from otter.dam.graph import *

with Graph() as graph:

    print(type(graph))

    a = Variable(np.array([[10, 15], [4, 8], [1, 2]]))
    b = Variable(np.array([[1, 2], [4, 8], [2, 4]]))
    c = Variable(np.array([[4, 5], [2, 13], [1, 8]]))
    d = Variable(np.array([[4, 3], [6, 10], [2, 10]]))

    e = a.sub(b)

    f = c.dot(d.T())

    g = f.dot(e)

    h = Variable(np.array([[1, 2],[3, 4],[5, 6]]))

    i = g.multiply(h)

    print(i.value)

    graph.update_gradient(i)

    print("a", a.gradient)
    print("b", b.gradient)
    print("c", c.gradient)
    print("d", d.gradient)
    print("e", e.gradient)
    print("f", f.gradient)
    print("g", g.gradient)
    print("h", h.gradient)
    print("i", i.gradient)

