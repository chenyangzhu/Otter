# Otter

Otter is a free open-source deep learning framework under development.

The main purpose of Otter is to test new ideas in implementing deep learning in
a more efficient and more friendly way.

## Useful Features

### Graph + Autogradient
In otter, you never need to do gradients yourself. As long as you have computed the forward propagation, Otter will automatically register all the calculation into a computation graph. It will then do the back propagation for your automatically with deep first graph search, as friendly as Python should have been.
```
with otter.Graph() as graph:
  a = Variable(np.array([[10, 15], [4, 8], [1, 2]]))
  b = Variable(np.array([[1, 2], [4, 8], [2, 4]]))

  c = a.add(b)
  graph.update_gradient(c)
```
