from otter.dam.structure import *
from otter.dam.graph import *

with Graph() as graph:

    print(type(graph))

    a = Variable(np.array([10, 15]))
    b = Variable(np.array([1, 2]))
    c = Variable(np.array([4, 5]))
    d = Variable(np.array([4, 5]))

    e = a.sub(b)
    f = c.sub(d)

    g = e.add(f)
    print(g.value)

    graph.update_gradient(g)

    print(a.gradient)
    print(b.gradient)
    print(c.gradient)
    print(d.gradient)
    print(e.gradient)
    print(f.gradient)
    print(g.gradient)