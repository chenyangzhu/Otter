# Otter

Otter is a free open-source deep learning framework under development.

The main purpose of Otter is to test new ideas in implementing deep learning in
a more efficient and more friendly way.

## Useful Features

### Graph + Autogradient
In Otter, you never need to do gradients yourself. As long as you have computed the forward propagation, Otter will automatically register all the calculation into a computation graph. It will then do the back propagation for your automatically with deep first graph search, as friendly as Python should have been.
```
with otter.Graph() as graph:
  a = Variable(np.array([[10, 15], [4, 8], [1, 2]]))
  b = Variable(np.array([[1, 2], [4, 8], [2, 4]]))

  c = a + b
  graph.update_gradient(c)
```

### Build a Deep Learning model from scratch
In otter, we provide friendly wrappers to create new deep learning models.
These layers are easily found in the `./layers` file. Here's a simple version
of the code. Using these built-in layers, you no longer have to write the complex
forward and backward propagation any more.
```
# Create some artificial data
n = 1000
p = 10
m = 1
x = Variable(np.random.normal(0, 1, (n, p)))
y = Variable(np.random.normal(0, 1, (n, m)))

# Build a computation graph
with Graph() as g:
    layer1 = Dense(output_shape=10, activation=sigmoid)
    layer2 = Dense(output_shape=m, activation=sigmoid)
    optimizer = GradientDescent(0.8)
    loss = mean_squared_error

    for _ in range(1000):
        a = layer1.train_forward(x)
        b = layer2.train_forward(a)
        c = loss(y, b)

        g.update_gradient_with_optimizer(c, optimizer)
```

### Ways to prevent vanishing/exploding gradients

- Gradient Clipping
- Safe exp, by clipping values
- safe inversion and safe log , by adding a small ε
