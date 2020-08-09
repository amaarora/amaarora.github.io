# Group Normalization 
1. TOC 
{:toc}

## Introduction
In this blog post today, we will look at [Group Normalization](https://arxiv.org/abs/1803.08494) and also:
- The drawback of Batch Normalization 
- How to overcome this problem? 
- What is `GroupNorm`? 
- How does `GroupNorm` compare to `InstanceNorm`, `LayerNorm` and `BatchNorm`? 
- Do a small experiment to apply `GroupNorm` + `Weight Standardization` to `Pets` dataset and compare performance to vanilla `ResNet` with `BatchNorm` 

[Batch Normalization](https://arxiv.org/abs/1502.03167) is used is most state-of-the art computer vision to stabilise training. **BN** normalizes the features based on the mean and variance in a mini-batch. This has helped improve model performance, reduced training times and also helped very deep models converge.

### Drawback of BatchNorm
Despite it's great success, since BN normalizes accross the batch dimension, it is required for BN to work with sufficiently large batch size (eg. 32). A small batch size leads to innacurate calculation of batch statistics and the model's error increases sufficiently. 

![](/images/BN_batch_size.png "fig-1 Imagenet classification error vs batch sizes")

As can be seen in the image above, the classification error when using **BN** is significantly higher for much smaller batch sizes (eg. 2). 

### How to overcome this problem
Now you might ask - "**But can't we always make sure that batch sizes are higher?**" **Synchronized BN** was introduced in [MegDet: A Large Mini-Batch Object Detector](https://arxiv.org/abs/1711.07240) whose mean and variance are computed accross multiple GPUs, thus, allowing larger batch sizes. However, this only migrates the problem to engineering and hardware demands and does not actually solve the problem. Also, the restriction on batch sizes is more demanding in computer vision tasks such as segmentation, video recognition, object detection and other high level systems built on them. For such applications it is not always possible to have a higher batch size. 

Also, this requirement of having larger batch sizes when using **BN** limits experimentation with much larger models (eg. efficient-net-B7). The usage of **BN** often requires a compromise between model design and batch sizes.

**Group Normalization (GN)** was proposed as a solution whose computation is independent of batch sizes. 

While there are also other existing methods like [Layer Normalization](https://arxiv.org/abs/1607.06450), [Instance Normalization](https://arxiv.org/abs/1607.08022), the **GN** paper reports that **GN** has better success for visual recognition tasks.

## Group Normalization

![](/images/GN_BN_LN_IN.png "fig-2 Normalization methods")

Above is an excellent representation of the difference between **BatchNorm**,  **LayerNorm**, **InstanceNorm** and **GroupNorm**.

Without delving much into the math as in the **GN** research paper, let's consider we have a batch of size `(N, C, H, W)` dimensions that needs to be normalized where, `N` represents batch size, `C` represents number of channels and `H` and `W` stand for height and width of the images in the batch.

In **BN**, the pixels sharing the same channel-index are normalized together. That is, **BN** normalizes accross the `(N,H,W)` axes. **LayerNorm**, normalizes accross the `(C,H,W)` axes for each sample.
**InstanceNorm**, normalizes along the `(H,W)` dimension for each sample and each channel.

### What does **Group Normalization** do?
**GN** first divides the channels intro groups. By default, there are 32 groups (from the research paper). So for a batch of dimension *(16, 64, 224, 224)*, there are 2 channels per group and **GN** calculates the mean and variance along a group of channels.

This idea of group wise computation is not new - [ResNext](https://arxiv.org/pdf/1611.05431.pdf) used the dimension of a group as a model design and called it 'cardinality'. Group Convolutions were also presented in [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) for distributing a model into two GPUs. There are also other examples in the literature like [SIFT](https://people.eecs.berkeley.edu/~malik/cs294/lowe-ijcv04.pdf), [HOG](https://link.springer.com/article/10.1023/A:1011139631724) and [GIST](https://ieeexplore.ieee.org/document/1467360) which are group-wise representations by design.


## GroupNorm in PyTorch
Following the snippet of code provided in the research paper:
```python
def GroupNorm(x, gamma, beta, G, eps=1e−5): 
    # x: input features with shape [N,C,H,W] 
    # gamma, beta: scale and offset, with shape [1,C,1,1] 
    # G: number of groups for GN
    N, C, H, W = x.shape 
    x = tf.reshape(x, [N, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep dims=True) 
    x = (x − mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [N, C, H, W]) 
    return x ∗ gamma + beta
```

We could rewrite this in `PyTorch` like so:

```python 
import torch
import torch.nn as nn

class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1,num_features,1,1))
        self.beta = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.num_groups ,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        # normalize
        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.gamma + self.beta
```

`PyTorch` inherent supports `GroupNorm` and can be used by using `nn.GroupNorm`.

## Does `GroupNorm` really work in practice?
Personally, I wanted to try a little experiment of my own to compare **GN** with **BN**. 

You can find the experiment in this notebook [here](https://nbviewer.jupyter.org/github/amaarora/amaarora.github.io/blob/master/nbs/Group%20Normalization%20WS.ipynb).

Basically, in the experiment, I trained a `resnet50` architecture on the `Pets` dataset. To my surprise, I found that simply replacing `BatchNorm` with `GroupNorm` led to sub-optimal results and the model with `GroupNorm` used as the normalization layer performed much worse than the model normalized with `BatchNorm` layer even for a very small batch size of 4. This was very different to the results reported in fig-1.

Thanks to [Sunil Kumar](https://twitter.com/DragonPG2000) who pointed me to [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370) research paper where I noticed that the researchers used a combination of [Weight Standardization](https://arxiv.org/abs/1903.10520) and **GN** to achieve SOTA results. So I tried this out with the implementation of Weight Standardization as in the official repository [here](https://github.com/joe-siyuan-qiao/WeightStandardization) and very quickly I was able to replicate the results with `GN+WS` performing significantly better than `BN` for batch size of 1.

The training logs for the notebook where I simply replaced **BN** with **GN** can be found [here](https://nbviewer.jupyter.org/github/amaarora/amaarora.github.io/blob/master/nbs/Group%20Normalization.ipynb).