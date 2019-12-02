# Otter

Otter is a free open-source deep learning framework under development.

The main purpose of Otter is to test new ideas in implementing deep learning in
a more efficient and more friendly way.

## Key Technical Bullet Points

#### Variables

Like TensorFlow and Pytorch, all variables are stored in either `tf.Variable` or `torch.tensor`. In Otter, 	`otter.Variable` is the fundamental data structure that still offers a great amount of interchangeablity with `np.array`. These Variables are like nodes that would later be connected on the computational graph.

#### New Computations
All computations are rewritten to match the new data type `otter.Variable`. The rewritten computation also includes a related back-computation. For example `otter.multiply` that does the Hadamard product, would also have an alternative `otter.back_multiply`, which along comes with the forward multiply.

#### Computation Graph
In Otter, all nodes are connected with linked lists. Once you add the first Variable to a graph, a new linked list is created. Then, if computations are being made on this Variable, a child node will be created, and the parent will be registered on the child node, along with the computation that brings this child.

#### Backpropagation
Backpropagations are done with ease in Otter. A simple deep first search on the linked lists would be just enough for a backpropagation. With optimization methods embedded into the deep first search, the backpropagation and gradient update could be performed in only one loop.

#### CNN optimization
Following the implementation in TensorFlow. The first step in Otter is to stretch the 4D `[n, channel, height, width]` picture into a 2D tensor, that has the dimension of `[n, channel x height x width]`. Then, we multiply this 2D tensor with the kernel of CNN. The kernel of CNN is often known as having several dimensions and layers `[number of kernels, channel, kernel_height, kernel_width]`. But such kernel can be stored as a 2D tensor, and what's more in Otter, it was stored as a sparse matrix to speed up computation. Then after this simple matrix multiplication, we can get a matrix of dimension `[n, channel x new_height x new_weight]`. All we have to do now is to reshape it into the outcome `[n, channel, new_height, new_weight]`, and the forward propagation is done. On the other hand, for backpropagation, we only have to do the exact same backpropagation for the matrix multiplication, which is really convenient.

#### Dropout Layer
A friend of mine was asked how to write Dropout Layer in a deep learning framework in his interview. My solution here in Otter is, to create a matrix at the same size of the input matrix. This matrix is randomly filled with 1/0, the rate we set in the Dropout Layer. Then everything went back to the simple element-wise matrix multiplication. Easy done;]


## Code samples

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
