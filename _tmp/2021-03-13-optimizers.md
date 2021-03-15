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

Now to implement `SGD` optimizer, we just need to create a method called `step` that does the optimization step and updates the value of the model parameters based on the gradients. 

```python 
class SGDOptimizer(Optimizer):
    def __init__(self, params, **defaults):
        super().__init__(params, **defaults)
    
    def step(self):
        for p in self.grad_params():
            p.data.add_(p.grad.data, alpha=-self.defaults['lr'])
```

This line `p.data.add_(p.grad.data, alpha=-self.defaults['lr'])` essentially does `p = p - lr * p.grad` which is the SGD step for each mini-batch. Thus, we have successfully re-implemented SGD Optimizer. 

To 

## SGD with Momentum

Classical Momentum as described in [this](http://www.cs.toronto.edu/~hinton/absps/momentum.pdf) paper can be defined as: 

![](/images/CM.png "eq-1 Classical Momentum")

Here `µ` represents the momentum factor, typically `0.9`. **Δ𝑓(θ<sub>t</sub>)** represents the gradients of parameters `θ` at time `t`. And `ε` represents the learning rate. 

As can be seen from `eq-1`, essentially we add a factor `µ` times the value of the previous step to the current step. Thus instead of going `p = p - lr * p.grad`, the new step value becomes `new_step = µ * previous_step + lr * p.grad` whereas previously for `SGD`, the step value was `lr * p.grad`.

Now to implement `SGD with Momentum`, we would need to be able to keep a track of the previous steps for each of the parameters. This can be done as below: 

```python
class SGDOptimizer(Optimizer):
    def __init__(self, params, **defaults):
        super().__init__(params, **defaults)
        self.lr = defaults['lr']
        self.µ  = defaults['momentum']
        self.state = defaultdict(dict)
    
    def step(self):
        for p in self.grad_params():
            param_state = self.state[p]
            
            d_p = p.grad.data            
            if 'moment_buffer' not in param_state:
                buf = param_state['moment_buffer'] = torch.clone(d_p).detach()
            else:
                buf = param_state['moment_buffer']
            
            buf.mul_(self.µ).add_(d_p)
            
            p.data.add_(buf, alpha=-self.lr)
```

From Sebastian Ruder's [blog](https://ruder.io/optimizing-gradient-descent/index.html#stochasticgradientdescent):
> At the core of `Momentum` is this idea - why don't we keep going in the same direction as last time? If the loss can be interpreted as the height of a hilly terrain, then the optimization process can then be seen as equivalent to the process of simulating the parameter vector as rolling on the landscape. Essentially, when using momentum, we push a ball down a hill. The ball accumulates momentum as it rolls downhill, becoming faster and faster on the way. The same thing happens to our parameter updates: The momentum term increases for dimensions whose gradients point in the same directions and reduces updates for dimensions whose gradients change directions. As a result, we gain faster convergence.

From a code implementation perspective, for each parameter inside `self.grad_params()`, we store a state called `momentum_buffer` that is initialized with the first value of `p.grad`. For every subsequent update, we do `buf.mul_(self.µ).add_(d_p)` which represents `buf = buf * µ + p.grad`. And finally, the parameter updates become `p.data.add_(buf, alpha=-self.lr)` which is essentially `p = p - lr * buf`. 

Thus, we have successfully re-implemented `eq-1`.

## RMSprop

`RMSprop` Optimizer brings to us an idea that why should all parameters have the step-size when clearly some parameters should move faster? It's great that `RMSprop` was actually introduced as part of a MOOC by Geoffrey Hinton in his [course](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).

From the PyTorch docs:
> The implementation here takes the square root of the gradient average before adding epsilon (note that TensorFlow interchanges these two operations). The effective learning rate is thus `α/(sqrt(v) + ϵ)` where `α` is the scheduled learning rate and `v` is the weighted moving average of the squared gradient.

The update step for `RMSprop` looks like:

![](/images/RMSprop.png "eq-2 RMSprop")

Essentially, for every parameter we keep a moving average of the Mean Square of the gradients. Next, we update the parameters similar to SGD but instead by doing something like `p = p - lr * p.grad`, we instead update the parameters by doing `p(t) = p(t) - (lr / MeanSquare(p, t)) * p(t).grad`.

Here, `p(t)` represents the value of the parameter at time `t`, `lr` represents learning rate and `MeanSquare(p, t)` represents the moving average of the Mean Square Weights of parameter `p` at time `t`.

> Key takeaway to be able to implement RMSprop - we need to able to store the exponentially weighted moving average of the mean square weights of the gradients. 

Therefore, we can update the implementation of `SGD` with momentum to instead implement `RMSprop` like so:

```python
class RMSPropOptimizer(Optimizer):
    def __init__(self, params, **defaults):
        super().__init__(params, **defaults)
        self.lr  = defaults['lr']
        self.α   = defaults['alpha']
        self.eps = defaults['epsilon']
        self.state = defaultdict(dict)
    
    def step(self):
        for p in self.grad_params():
            param_state = self.state[p]
            
            d_p = p.grad.data   
            if 'exp_avg_sq' not in param_state:
                exp_avg_sq = param_state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            else:
                exp_avg_sq = param_state['exp_avg_sq']
            
            exp_avg_sq.mul_(self.α).addcmul_(d_p, d_p, value=1-self.α)
            denom = exp_avg_sq.sqrt().add_(self.eps)
            
            p.data.addcdiv_(d_p, denom, value=-self.lr)
```

As can be seen inside the `step` method, we iterate through the parameters with gradients, and store the initial value of the gradients inside the a variable called `d_p` which represents derivative of parameter `p`. 

Next, we initialize the exponential moving average of the square of the gradients `exp_avg_sq` as an empty array filled with zeros of the same shape as `d_p`. For every next step, this `exp_avg_sq` is updated by this line of code: `exp_avg_sq.mul_(self.α).addcmul_(d_p, d_p, value=1-self.α)`. This equates to `exp_avg_sq = (self.α * exp_avg_sq)  + (1 - self.α * (d_p**2))`.

Therefore, we are keeping an exponentially weighted moving average of the square of the gradients. But as can be seen in `eq-2`, the update step of `RMSprop` actually divides by the `sqrt` of this `exp_avg_sq`. So our denominator `denom` becomes `exp_avg_sq.sqrt().add_(self.eps)`. `eps` is added for numerical stability. 

Finally, we do our update step `p.data.addcdiv_(d_p, denom, value=-self.lr)` which equates to `p = p - (self.lr * d_p)/denom` thus performing the `RMSprop` update step as in `eq-2`. 

Therefore, we have successfully re-implemented `RMSprop` from scratch. 