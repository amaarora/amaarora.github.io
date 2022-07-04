# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

1. TOC 
{:toc}

## Personal Update
For someone who was actively releasing blogs almost all throughout 2020 & 2021, I am kinda sad to admit that this is my first blog for the year 2022. But, at the same time, I am super excited to be back. My personal responsibilities took priority for the last 1 year and I had to give up on releasing blog posts. Now that the storm has settled, I am happy to be back. 

I also resigned from my position as **Machine Learning Engineer** from **Weights and Biases (W&B)** earlier this year and have joined [REA Group](https://www.realestate.com.au/) as **Data Science Lead**. It's quite a change in my day to day work life, but I am up for the challenge and enjoying every second of my new job so far. :) 

I wrote many blogs on various different research papers during my time at W&B that can be found [here](https://amaarora.github.io/). 

A lot has changed in the past 1 year or so since I have been away. As I catch-up with the latest research, I hope to continue releasing more blog posts and take you on this journey with me as well. Let's learn together! 

## Prerequisites 
As part of this blog post I am going to assume that the reader has a basic understanding of CNNs and the Transformer architecture. 

Here are a couple good resources on the Transformer architecture if you'd like some revision:
1. [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) 
2. [Vision Transformer (ViT)](https://amaarora.github.io/2021/01/18/ViT.html)

For CNNs, there are various architectures that have been introduced. I have previously written blogs about a few of them:
1. [Squeeze and Excitation Networks](https://amaarora.github.io/2020/07/24/SeNet.html)
2. [DenseNet](https://amaarora.github.io/2020/08/02/densenets.html)
3. [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://amaarora.github.io/2020/08/13/efficientnet.html)

## Introduction
As part of today's blog post, I want to cover [Swin Transformers](https://arxiv.org/abs/2103.14030). As is usual for my blog posts, I will be covering every related concept in theory along with a working PyTorch implementation of the architecture from [TIMM](https://github.com/rwightman/pytorch-image-models). Also, all text presented in this blog post directly from the paper will be in *Italics*.

> **NOTE**: At the time of writing this blog post, we already have a V2 of the [Swin Transformer](https://arxiv.org/abs/2111.09883) architecture. This architecture will be covered in a future blog post. 

While the Transformer architecture before this paper had proved to be performing better than CNNs on the ImageNet dataset, it was yet to be utilised as a general purpose backbone for other tasks such as object detection & semantic segmentation. This paper solves that problem and Swin Transformers can capably serve as general purpose backbones for computer vision. 

From the Abstract of the paper: 

*Swin Transformer is compatible for a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO testdev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures.*

## Key Concepts/Ideas
I might be oversimplifying here, but in my head there are only two new key concepts that we need to understand on top of ViT to get a complete grasp of the Swin Transformer architecture. 

1. Shifted Window Attention 
2. Patch Merging

Everything else to me looks pretty much the same as ViT (with some minor modifications). So, what are the two concepts? We will get to them later in this blog post. 

First, let's get a high level overview of the architecture. 

## Swin Transformer Overview

![](/images/swin-transformer.png "Swin Transformer Architecture")

From section 3.1 of the paper: 

*An overview of the Swin Transformer architecture is presented in the Figure above, which illustrates the tiny version (SwinT). It first splits an input RGB image into non-overlapping patches by a patch splitting module, like ViT. Each patch is
treated as a “token” and its feature is set as a concatenation of the raw pixel RGB values. In our implementation, we use a patch size of 4 × 4 and thus the feature dimension of each patch is 4 × 4 × 3 = 48. A linear embedding layer is applied on this raw-valued feature to project it to an arbitrary dimension.*

### Patch Partition/Embedding
So first step is to take in an input image and convert it to Patch Embeddings. This is the exact same as ViT with the difference being that each patch size in Swin Transformer is *4 x 4* instead of *16 * 16* as in ViT. I have previously explained Patch Embeddings [here](https://amaarora.github.io/2021/01/18/ViT.html#patch-embeddings) and therefore won't be going into detail here. A PyTorch implementation of Patch Embeddings looks like (presented below for sake of completeness): 

```python 
from torch import nn as nn
from timm.helpers import to_2tuple
from timm.trace_utils import _assert

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
```

I won't be going into detail here and would kindly ask the reader to refer to my [previous blog post](https://amaarora.github.io/2021/01/18/ViT.html#patch-embeddings) for an explaination of this code. 
