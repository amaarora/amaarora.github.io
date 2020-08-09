# Group Normalization 
1. TOC 
{:toc}

## Introduction
In this blog post today, we will look at [Group Normalization](https://arxiv.org/abs/1803.08494) research paper and also:
- The drawback of [Batch Normalization](https://arxiv.org/abs/1502.03167)  
- Introduction to **Group Normaliation**
- Other normalization techniques and how does **Group Normalization** compares to those
- Benefits of **Group Normalization** over other techniques
- Discuss **Group Division** and 32 as default number of groups
- Discuss effect of **Group Normalization** on deeper models (eg. Resnet-101)
- Analyse how **Group Normalization** and **Batch Normalization** are qualitatively similar
- Implement **Group Normalization** in *PyTorch* and *Tensorflow*
- Do a small experiment to apply **GroupNorm** + **Weight Standardization** to **Pets** dataset and compare performance to vanilla **ResNet** with **BatchNorm** 

[Batch Normalization](https://arxiv.org/abs/1502.03167) is used is most state-of-the art computer vision to stabilise training. **BN** normalizes the features based on the mean and variance in a mini-batch. This has helped improve model performance, reduced training times and also helped very deep models converge.

But this technique also suffers from drawbacks - if batch size is too small, training becomes unstable with BN. 

The aim is to not study BN, many other wonderful posts have been written on that, but to look at other alternatives.

Through this blog post, I hope to introduce **Group Normalization** as an alternative to **Batch Normalization** and help the reader develop an intuition for cases where GN could perform better than BN.

### Drawback of Batch Normalization
Knowingly or unknowingly, we have all used BN in our experiments. If you have trained a `ResNet` model or pretty much any other CV model using *PyTorch* or *Tensorflow*, you have made use of BN to normalize the deep learning network.

From the Group Normalization research paper,
> We all know that BN has been established as a very effective component in deep learning. BN normalizes the features by the mean and variance computed within a batch. But despite its great success, BN exhibits drawbacks that are also caused by its distinct behavior of normalizing along the batch dimension. In particular, it is required for BN to work with sufficiently large batch size. A small batch size leads to innacurate estimation of the batch statistics and reducing BN's batch size increases the model error dramatically.

### Introduction to Group Normalization
In the paper, the authors introduce GN as a simple alternative to BN. From the paper:

> GN divides the channels into groups and computes within each group the mean and variance for normalization. GN's computation is independent of batch sizes, and its accuracy is stable in a wide range of batch sizes. 

Essentially, GN takes away the dependance on batch size for normalization and in doing so mitigates the problem suffered by BN. There are also other techniques that have been proposed to avoid batch dimension - but we will discuss them later. For now, it is essential for the reader to realize, that instead of normalizing accross the batch dimension, GN normalizes accross the groups (channel dimension).

![](/images/BN_batch_size.png "fig-1 Imagenet classification error vs batch sizes")

As can be seen in the image above, because GN does not depend on the batch size, the classification error when the deep learning model is normalized using GN is stable accross various batch sizes compareed to BN. 

![](/images/GN_bs_2.png "fig-2 ResNet-50's validation eror trained with bs 32, 16, 8, 4 and 2")

The same trend as `fig-1` can also be observed in `fig-2` where the validation error is consistent accross various batch sizes for GN as opposed to BN. Another key thing to note, the validation error for GN as reported in the research paper is very similar to that for BN - therefore, GN can be considered as a strong alternative to BN. 

The validation errors (from the research paper) are presented in `table-1` below:

![](/images/bs_sensitivity_gn.png "table-1 Sensitivity to batch sizes")


While BN performs slightly better than GN for batch size 32, GN performs better for all lower batch sizes. 

## Other Normalization Techniques
`Group Normalization` isn't the first technique that was proposed to overcome the drawback of BN. There are also several other techniques such as [Layer Normalization](), [Instance Normalization]() and others mentioned in the references of this blog post. 

But, GN is the first technique to achieve comparable validation error rates as compared to BN. 

In this section we look at the most popular normalization tecniques namely - Layer Normalization (LN), Instance Normalization (IN), Batch Normalization (BN) and Group Normalization (GN).

### BatchNorm, GroupNorm, InstanceNorm and LayerNorm

![](/images/GN_BN_LN_IN.png "fig-3 Normalization methods")

The above image presented in the research paper is one of the best ways to compare the various techniques and get an intuitive understanding. 

Let's consider that we have a batch of dimension `(N, C, H, W)` that needs to be normalized. 

Here, 
- `N`: Batch Size
- `C`: Number of Channels
- `H`: Height of the feature map
- `W`: Width of the feature map

Essentially, in **BN**, the pixels sharing the same channel index are normalized together. That is, for each channel, **BN** computes the *mean* and *std deviation* along the `(N, H, W)` axes. As we can see, the group statistics depend on `N`, the batch size. 

In **LN**, the *mean* and *std deviation* are computed for each sample along the `(C, H, W)` axes. Therefore, the calculations are independent of the batch size. 

In **IN**, the *mean* and *std deviation* are computed for each sample and each channel along the `(H, W)` axes. Again, the calculations are independent of batch size. 

Finally, for group norm, the batch is first divided into groups (32 by default, discussed later). The batch with dimension `(N, C, W, H)` is first reshaped to `(N, G, C//G, H, W)` dimensions where `G` represents the **group size**. Finally, the *mean* and *std deviation* are calculated along the `(H, W)` and along `C//G` channels. This is also illustrated very well in `fig-3`.

One key thing to note here, if `C == G`, that is the number of groups are set to be equal to the number of channels (one channel per group), then **GN** becomes **IN**. 

And if, `G == 1`, that is number of groups is set to 1, **GN** becomes **LN**. 

I would like for the reader to take a minute here and make sure that he understands the differences between these techniques mentioned above.

### Benefits of Group Normalization over other techniques

Also, it is important to note that **GN** is less restricted than **LN**, because in **LN** it is assumed that all channels in a layer make "equal contributions" whereas **GN** is more flexible because in **GN**, each group of channels (instead of all of them) are assumed to have shared mean and variance - the model still has flexibility of learning a different distribution for each group. 

Also, **GN** is slightly better than **IN** because **IN** normalizes accross each sample for each channel, therefore, unlike **GN**, it misses the opportunity of exploiting the channel dependence.

![](/images/gn_comp.png "fig-4 Comparison of error curves")

Therefore, due to the reasons discussed above, we can see that the validation and training errors for **GN** are lower than those for **LN** and **IN**.

## Implementation of GroupNorm
The following snippet of code has been provided in the research paper:
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

## Does GroupNorm really work in practice?
Personally, I wanted to try a little experiment of my own to compare **GN** with **BN**. 

You can find the experiment in this notebook [here](https://nbviewer.jupyter.org/github/amaarora/amaarora.github.io/blob/master/nbs/Group%20Normalization%20WS.ipynb).

Basically, in the experiment, I trained a `resnet50` architecture on the `Pets` dataset. To my surprise, I found that simply replacing `BatchNorm` with `GroupNorm` led to sub-optimal results and the model with `GroupNorm` used as the normalization layer performed much worse than the model normalized with `BatchNorm` layer even for a very small batch size of 4. This was very different to the results reported in fig-1.

Thanks to [Sunil Kumar](https://twitter.com/DragonPG2000) who pointed me to [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370) research paper where I noticed that the researchers used a combination of [Weight Standardization](https://arxiv.org/abs/1903.10520) and **GN** to achieve SOTA results. So I tried this out with the implementation of Weight Standardization as in the official repository [here](https://github.com/joe-siyuan-qiao/WeightStandardization) and very quickly I was able to replicate the results with `GN+WS` performing significantly better than `BN` for batch size of 1.

The training logs for the notebook where I simply replaced **BN** with **GN** can be found [here](https://nbviewer.jupyter.org/github/amaarora/amaarora.github.io/blob/master/nbs/Group%20Normalization.ipynb).