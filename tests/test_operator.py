from otter.dam.structure import *
from otter.dam.graph import *
from otter.activation import sigmoid


with Graph() as graph:

    a = Variable(np.array([[0, 2], [4, 8], [1, 2]]))
    b = Variable(np.array([[1, 2], [4, 8], [2, 4]]))
    h = Variable(np.array([[4, 5], [2, 4], [1, 8]]))
    i = Variable(np.array([[4, 3], [6, 5], [2, 1]]))

    c = a.add(b)
    j = b.sub(h)
    e = c.dot(j.T())
    f = j.dot(i.T())
    g = e.sub(f)
    h = g ** 2
    graph.update_gradient(h)

    print("a", a.gradient)
    print("b", b.gradient)
    print("c", c.gradient)
    # print("d", d.gradient)
    print("e", e.gradient)
    print("f", f.gradient)
    print("g", g, g.gradient)
    print("h", h.gradient)
    print("i", i.gradient)
    print("j", j.gradient)
