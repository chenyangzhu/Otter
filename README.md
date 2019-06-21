# Otter

Otter is a free open-source deep learning framework under development.

The main purpose of Otter is to test new ideas in implementing deep learning in
a more efficient and more friendly way.

## Useful Features

### Autogradient
In Otter, you never need to do gradients yourself. As long as you have computed the forward propagation, Otter will automatically register all the calculation into a computation graph. It will then do the back propagation for your automatically with deep first graph search, as friendly as Python should have been.
```
a = Variable(np.array([[10, 15], [4, 8], [1, 2]]))
b = Variable(np.array([[1, 2], [4, 8], [2, 4]]))

c = a + b
c.back_propagation()
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
        a = layer1.forward(x)
        b = layer2.forward(a)
        c = loss(y, b)

        g.update_gradient_with_optimizer(c, optimizer)
```

### Ways to prevent vanishing/exploding gradients

- Gradient Clipping
- Safe exp, by clipping values
- safe inversion and safe log , by adding a small ε

### Other features

- load and save models with json
- Flexible Model Structures (beta)

### What's next
- Optimization Methods including Adam, etc.
// 过去的时候去n，回来的时候只回1，才可以加速
- Faster computations on CNN
- etc.
