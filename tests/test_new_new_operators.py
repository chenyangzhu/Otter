from otter.dam.structure import *
from otter.dam.graph import *
from otter.activation import sigmoid
from otter.optimizer import GradientDescent

with Graph() as graph:

    print(type(graph))

    a = Variable(np.array([[10, 15], [4, 8], [1, 2]]))
    b = Variable(np.array([[1, 2], [4, 8], [2, 4]]))
    h = Variable(np.array([[4, 5], [2, 13], [1, 8]]))
    i = Variable(np.array([[4, 3], [6, 10], [2, 10]]))

    c = a.add(b)
    d = h.sub(i)

    e = c.dot(d.T())
    f = e.pow(2)

    f.back_propagation()
    # graph.update_gradient(f)
    print("a", a.gradient)
    print("b", b.gradient)
    print("c", c.gradient)
    print("d", d.gradient)
    print("e", e.gradient)
    # print("f", f.gradient)
    # ("g", g.gradient)
    print("h", h.gradient)
    print("i", i.gradient)
    # print("j", j.gradient)