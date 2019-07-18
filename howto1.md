# How to write your own deep learning framework? - Computation Graph

## Introduction
TensorFlow and Pytorch are two very famous deep learning frameworks that work
pretty well on almost all platforms. With C/C++ and CUDA backends, they perform
almost the fastest among all others. Is this the end of deep learning frameworks?
Of course not. In this article and the following several articles, we'll go through some of the most important features of these frameworks and see if we can, to some extent,
make things better.

In this article, we'll talk about computation graph and auto-grad algorithms. We'll cover how they works and how this idea blends in graph theories. Computation graphs are the most important element in Deep Learning frameworks, as explicitly stated in TensorFlow and implicitly in PyTorch. For me, I started to recognize the computation graph as merely a math result from chain rules, but simply see the computation graph as chain rules
might not be enough for us to write out the framework. Let's dig a bit deeper into
this concept. First, let's talk about variables or tensors.

My simple framework is called otter and is published on github https://github.com/chenyangzhu/otter. Feel free to check it out.


### Tensors and Operators
Why are variables/tensors important? (to make things easier for me, I'll call them tensors from now, though they essentially mean different things in math.). These are the elements in your deep learning framework. Recall that the fundamental interest in Deep Learning is to approximate a very complex function using only the input and the output. These inputs and outputs are all tensors. All the parameters you use are tensors too.

What deep learning frameworks are doing is that they bridge these variables with operators like addition, concatenation and maximum. By using one operator on a tensor,
we then have a new tensor.

This concept is essentially important in that
- Any complex math functions, including softmax or cross-entropy can be decomposed into simple operators.
- All operators accept 1-2 inputs and return 1 output. For example, addition requires two tensors (a + b) and maximum only requires one (max(A)). This is similar to the idea of a binary tree, where we have two children and one parent.
- Any tensor can be the parent of only 1-2 variables, but it could be the child of many others.

The first step of the deep learning framework, is to create these tensors. In tensorflow and pytorch, they are written in C, but you don't have to if you just want to have some fun with deep learning. You can well use numpy as your backend and never need to worry about the C stuff. But you should notice that numpy is way slower than C backends and your framework might never be faster than Pytorch or TensorFlow. We'll talk about how
pytorch store their tensors in later articles. Now, we only need to focus on the concept of tensors and operators.

After creating the tensors, we now need to write the operators. You might wonder why
we don't use the numpy operators or the native python operators. It makes little sense if you only consider forward propagation, but when it comes to backward propagation, it would be much more convenient if we write a new operator and its corresponding back_operator.

The back_operator is just performing the backward propagation only on this operator.
For example, when the gradient is back-propagated to this variable. Then we need to decide how we want to backprop this the incoming gradient to its two children. This
makes a new back_operator essentially important.

In otter, these parts are written in `otter/dam/strucure.py`.

### Computation Graph
Once we have these tensors and operators, we can easily build up the computation graph.
Computation graphs can be decomposed into essentially binary tree structures. This
concept works very well if we now introduce forward and backward propagation.

- Forward Propagation. Performing any operators (forward propagation) is the same as adding a parent for one tensor. Not a very difficult concept to understand.
- Backward Propagation. Backward propagation is the same as deep first search from parent. Every time we do a back propagation, we start from the output variable, we initialize its variable, and we perform back_operator to feed the gradient to their children.

In otter, these parts are written in `otter/dam/graph.py`.

### Conclusions
In this article, we briefly discussed how to write tensors and graphs in deep learning and how they are important in deep learning frameworks. In the next article, if any, I will talk about how to fast compute CNN's.
