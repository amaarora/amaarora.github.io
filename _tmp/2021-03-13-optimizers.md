# "Adam" and friends

Who's Adam? Why should I care about ("it's" or "his") friends?!

[Adam](https://arxiv.org/abs/1412.6980) is an `Optimizer`. He has many friends but his dearest are [SGD](http://www.cs.toronto.edu/~hinton/absps/momentum.pdf), Momentum & [RMSprop](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).

In this blog post we are going to meet "Adam" and his friends and get to know about them in a lot more detail!

Each of Adam's friends has contributed to Adam's personality. So to get to know Adam very well, we should first meet the friends. We start out with `SGD` first, then meet `Momentum`, `RMSprop` and finally `Adam`. 

1. TOC 
{:toc}

## Introduction 
In this blog post we are going to re-implement `SGD`, `Momentum`, `RMSprop` and `Adam` from scratch. [Jeremy Howard](https://twitter.com/jeremyphoward) has already shown how [fastai](http://docs.fast.ai/) implements these algorithms building on top of a "generic" Optimizer from scratch [here](https://youtu.be/hPQKzsjTyyQ?t=4188).

[Here's](https://youtu.be/CJKnDu2dxOE?t=6235) another video by Jeremy that implements these algorithms in **Microsoft Excel**! 

In this blog post, the code for the Optimizers has been mostly copied from [PyTorch](https://pytorch.org/) but follows a different structure to keep the code implementations to a minimum. The implementations for these various Optimizers in this blog post are "much shorter" than those in PyTorch.

> This blog post aims at helping you get an intuition about these various different Optimization algorithms including the code implementations. This blog post follows a more practical approach rather than a theoritical one and we understand about these algorithms more from an implementation and experimentation perspective. 

By the end of this blog post we should be able to compare the performance of our implementations with PyTorch's implementations as below: 

![](/images/optimizers.png "fig-1 Adam and Friends")

## Prerequisite
A wonderful introduction to Optimization has been presented [here](https://cs231n.github.io/optimization-1/#optimization). 

> In this blog post we will not introduce Optimization and it is assumed that the reader has some prior experience with Optimizers or a general understanding of what Optimizers do. In general there are many great resources that are already present that introduce Optimization & SGD. These can be found in the [Resources]() section of this blog post. 

In [this](https://youtu.be/ccMHJeQU4Qw?t=4575) video, Jeremy implements SGD from scratch and explains all the details with an introduction to Optimizers. 

It is also assumed that the reader has some knowledge about PyTorch and has previously used PyTorch in some form to train neural networks. If not, then the above video by Jeremy is again a great introduction. 

## Stochastic Gradient Descent
In this section we will first introduce what is Stochastic Gradient Descent and then based on our understanding, implement it in PyTorch from scratch. 

### What is Stochastic Gradient Descent?
From the introductions shared in the Prerequisite section, you might already know that to perform Gradient Descent, we need to be able to calculate the gradients of some function that we wish to minimise with respect to the parameters. We don't need to manually calculate the gradients and as mentioned in [this](https://youtu.be/ccMHJeQU4Qw?t=4575) video by Jeremy, PyTorch can already do this for us using [torch.autorgrad](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html).

So now that we know that we can compute the gradients, the procedure of repeatedly evaluating the gradient and then performing a parameter update is called `Gradient Descent`. Its vanilla version looks as follows:

```python
# Vanilla Gradient Descent

for epoch in range(num_epochs):
    predictions = model(training_data)
    loss = loss_function(predictions, ground_truth)
    weights_grad = evaluate_gradient(loss) # using torch by calling `loss.backward()`
    weights -= learning_rate * weights_grad # perform parameter update
  ```

In Vanilla Gradient Descent, we first get the predictions on the whole training data, then we calculate the loss using some loss function. Finally, we update the weights in the direction of the gradients to minimise the loss. We do this repeatedly for some predefined number of epochs. 

> Can you think of possible problems with this approach? Can you think of why this approach could be computationally expensive? 

In large-scale applications, the training data can have on order of millions of examples. Hence, it seems wasteful to compute the full loss function over the entire training set in order to perform only a single parameter update. A very common approach to addressing this challenge is to compute the gradient over batches of the training data. This approach is reffered to as `Stochastic Gradient Descent`:

```python 
# Vanilla Stochastic Gradient Descent
for epoch in range(num_epochs):
    for input_data, labels in training_dataloader:
        preds = model(input_data)
        loss  = loss_function(preds, labels)
        weights_grad = evaluate_gradient(loss) # using torch by calling `loss.backward()`
        weights -= learning_rate * weights_grad # perform parameter update
```

In `Stochastic Gradient Descent`, we divide our training data into sets of batches. This is essentially what the [DataLoader](https://pytorch.org/docs/stable/data.html) does, it divides the complete training set into batches of some predefined `batch_size`.

So let's keep the key things in our mind before we set out to implement SGD: 
1. Divide the training data into batches, PyTorch DataLoaders can do this for us. 
2. For each mini-batch:
    - Make some predictions on the input data and calculate the loss.
    - Calculate the gradients using `torch.autograd` based on the loss. 
    - Take a step in the opposite direction of gradients to minimise the loss. 


### `SGD` Code Implementation
We follow a similar code implementation to PyTorch. In PyTorch as mentioned [here](https://pytorch.org/docs/stable/optim.html), there is a base class for all optimizers called `torch.optim.Optimizer`. It has some key functions methods like `zero_grad`, `step` etc. Remember from our general understanding of SGD, we wish to be able to update the parameters (that we want to optimize) by taking a step in the opposite direction of the gradients to minimise the loss function. 

Thus, from a code implementation perspective, we would need to be able to iterate through all the `parameters` and do `p = p - lr * p.grad`, where `p` refers to parameters and `lr` refers to learning rate. 

With this basic understanding let's implement an Optimizer class below:

```python 
class Optimizer(object):
    def __init__(self, params, **defaults):
        self.params = list(params)
        self.defaults = defaults
    
    def grad_params(self):
        return [p for p in self.params if p.grad is not None]
    
    def step(self): 
        raise NotImplementedError
    
    def zero_grad(self):
        for p in self.grad_params():
            p.grad.zero_()
```

The `Optimizer` class above implements two main methods - `grad_params` and `zero_grad`. Doing something like `self.grad_params()` grabs all those parameters as a list whose gradients are not None. Also, calling the `zero_grad()` method would zero out the gradients as explained in [this](https://youtu.be/ccMHJeQU4Qw?t=4575) video.

---
> At this stage you might ask, what are these parametrs? In PyTorch calling `model.parameters()` returns a generator through which we can iterate through all parameters of our model. A typical training loop as you might have seen in PyTorch looks something like:
```python
model = create_model()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for input_data, labels in train_dataloader:
    preds = model(input_data)
    loss  = loss_fn(preds, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```
---

In the training loop above we first create an optimizer by passing in `model.parameters()` which represents the parameters that we wish to optimize. We also pass in a learning rate that represents the step size. In PyTorch, calling `loss.backward()` is what appends an attribute `.grad` to each of the parameters in `model.parameters()`. Therefore, in our implementation, we can grab all those parameters whose gradients are not None by doing something like `[p for p in self.params if p.grad is not None]`.