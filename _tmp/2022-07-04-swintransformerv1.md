# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

1. TOC 
{:toc}

## Personal Update
For someone who was actively releasing blogs almost all throughout 2020 & 2021, I am kinda sad to admit that this is my first blog for the year 2022. But, at the same time, I am super excited to be back. My personal responsibilities took priority for the last 1 year and I had to give up on releasing blog posts. Now that the storm has settled, I am happy to be back. 

I also resigned from my position as **Machine Learning Engineer** from **Weights and Biases (W&B)** earlier this year and have joined [REA Group](https://www.realestate.com.au/) as **Data Science Lead**. It's quite a big change in my day to day work life, but I am up for the challenge and enjoying every second of my new job so far. :) 

I wrote many blogs on various different research papers during my time at W&B that can be found [here](https://amaarora.github.io/). 

A lot has changed in the past 1 year or so since I have been away. As I catch-up with the latest research, I hope to continue releasing more blog posts fortnightly and take you on this journey with me as well. Let's learn together! 

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
As part of today's blog post, I want to cover [Swin Transformers](https://arxiv.org/abs/2103.14030). As is usual for my blog posts, I will be covering every related concept in theory along with a working PyTorch implementation of the architecture from [TIMM](https://github.com/rwightman/pytorch-image-models). Also, all text presented in this blog post copied directly from the paper will be in *Italics*.

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
treated as a “token” and its feature is set as a concatenation of the raw pixel RGB values. In our implementation, we use a patch size of 4 × 4 and thus the feature dimension of each patch is 4 × 4 × 3 = 48. A linear embedding layer is applied on this raw-valued feature to project it to an arbitrary dimension (denoted as C).*

### Patch Partition/Embedding
So first step is to take in an input image and convert it to Patch Embeddings. This is the exact same as ViT with the difference being that each patch size in Swin Transformer is **$4 x 4$** instead of **$16 x 16$** as in ViT. I have previously explained Patch Embeddings [here](https://amaarora.github.io/2021/01/18/ViT.html#patch-embeddings) and therefore won't be going into detail here.  

```python 
from timm.models.layers import PatchEmbed
x = torch.randn(1, 3, 224, 224)
patch_embed = PatchEmbed(img_size=224, patch_size=4, embed_dim=96)
patch_embed(x).shape

>> torch.Size([1, 3136, 96])
```

As can be seen, that the output of the Patch Embedding layer is of shape $1 x 3136 x 96$, that is, $1 x (H/4 x W/4) x 96$ where *$96$* is the embedding dimension. 

> NOTE: The embedding dimension 96 has been mentioned in section 3.3 of the paper under **Architecture Variants**.

### Stage-1 Overview
Continuing from section 3.1 of the paper:

*Several Transformer blocks with modified self-attention computation (Swin Transformer blocks) are applied to these patch tokens. The Transformer blocks maintain the number of tokens H/4 × W/4, and together with the linear embedding are referred to as “Stage 1”.*

```python 
stage_1 = BasicLayer(dim=96, out_dim=192,
                     input_resolution=(56, 56),
                     depth=2)
inp = torch.randn(1, 56*56, 96)
stage_1(inp).shape

>> torch.Size([1, 3136, 96])
```

As can be seen from the code snippet above, there is no change in the dimension of the input as it passes through "Stage-1". In fact, the dimension of the inputs doesn't change as it passes through every stage. It is in between stages, that a **Patch Merging** layer is applied to reduce the number of tokens as the network get's deeper. 

### Patch Merging Layer
From section 3.1 of the paper:

*The first patch merging layer concatenates the features of each group of 2×2 neighboring patches, and applies a linear layer on the 4C - dimensional concatenated features. This reduces the number of tokens by a multiple of 2×2 = 4 (2× downsampling of resolution), and the output dimension is set to 2C.*

Here, $C$ is the number of channels (embedding dimension). For the tiny version that has been explained as part of this blog post, $C=96$.

![](/images/PatchMerging.png "Patch Merging")

As can be seen below, the Patch-Merging layer merges four patches. So with every merge, both height and width of the image are further reduced by a factor of 2. In stage-1, the input resolution is $(H/4,W/4)$, but after patch merging, the resolution will change to $(H/8, W/8)$ which will be the input for stage-2. For stage-3 the input resolution would be $(H/16, W/16)$ and for stage-4, the input resolution would be $(H/32, W/32)$.

Let's understand the inputs and outputs for PatchMerging in code:

```python 
from timm.models.swin_transformer import PatchMerging
x = torch.randn(1, 56*56, 96)
l = PatchMerging(input_resolution=(56, 56), dim=96, out_dim=192, norm_layer=nn.LayerNorm)
l(x).shape
```

As can be seen, the output width and height are both reduced by a factor of /2 and the number of output channels is 2 x C where C is the number of input channels, here for Swin-T, $C=96$.

Let's look at the source code for Patch Merging now that we understand it's functionality: 

```python 
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)

    def forward(self, x):
        """
        x: B, H*W, C
        B: Batch size 
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C      
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        return x
```

To help understand the code above, I used Microsoft Excel again. In the figure below, 
    
- $X_0$ is represented by `✔️` - starting at row 0, column 0;
- $X_1$ is represented by `❌` - starting at row 1, column 0;
- $X_2$ is represented by `⚫` - starting at row 0, column 1;
- $X_0$ is represented by `⬛` - starting at row 1, column 1 

<img src="../images/patch-merging-excel.png" alt="Patch Merge" width="600"/>

Therefore, when we concatenate in code using `x = torch.cat([x0, x1, x2, x3], -1)`, we are actually merging four patches together, and therefore here $X$ would have dimension size of 4C. Next, as was mentioned in the paper - *the output dimension is set to 2C*, therefore, we make use of a `nn.Linear` layer in code to reduce the dimension side to 2C.

Now that we've looked at Patch Merging layer, let's get to the meat of the paper - which is the **Swin Transformer Block**.

## Swin Transformer Block
At every stage in *Swin-T*, there are at two Swin Transformer Blocks except Stage-3, where there are 6 Swin Transformer Blocks in tandem. 

![](/images/swin-transformer-block.png "Two successive Swin Transformer Blocks")

From section Swin Transformer Block heading under section 3.1 of the paper:
*Swin Transformer is built by replacing the standard multi-head self attention (MSA) module in a Transformer block by a module based on shifted windows, with other layers kept the same. As illustrated in Figure above, a Swin Transformer block consists of a shifted window based MSA module, followed by a 2-layer MLP with GELU nonlinearity in between. A LayerNorm (LN) layer is applied before each MSA module and each MLP, and a residual connection is applied after each module.*

Okay, so pretty much everything is the same as ViT except this idea of shifted windows based attention. So, that's what we should really at next before looking at the code implementation of Swin Transformer Block.

### Shifted Windows based Self Atention
From section 3.2 of the paper:
*The standard Transformer architecture and its adaptation for image classification both conduct global selfattention, where the relationships between a token and all other tokens are computed. The global computation leads to quadratic complexity with respect to the number of tokens, making it unsuitable for many vision problems requiring an immense set of tokens for dense prediction or to represent a high-resolution image.*

*For efficient modeling, we propose to compute self-attention within local windows. The windows are arranged to evenly partition the image in a non-overlapping manner.*

*The window-based self-attention module lacks connections across windows, which limits its modeling power. To introduce cross-window connections while maintaining the efficient computation of non-overlapping windows, we propose a shifted window partitioning approach which alternates between two partitioning configurations in consecutive Swin Transformer blocks.*

*As illustrated in Figure below, the first module uses a regular window partitioning strategy which starts from the top-left pixel, and the 8 × 8 feature map is evenly partitioned into 2 × 2 windows of size 4 × 4 (M = 4). Then, the next module adopts a windowing configuration that is shifted from that of the preceding layer, by displacing the windows by $([M/2], [M/2])$ pixels from the regularly partitioned windows.

![](/images/shifted-windows.png "An illustration of the shifted window approach for computing self-attention in the proposed Swin Transformer architecture.")

Before looking at the code implementation of shifted window based attention, we first need to understand what's exactly going on. And if the above doesn't make much sense, let me try and break it down for you. It's really simple. Trust me!

So on the left, we have an 8x8 feature map which is evenly partitioned into 4 windows of size 4 x 4. Here, the window size $M=4$. Now in the first part of the two successive blocks, we calculate attention inside these windows. But, remember we also need cross-window attention for our network to learn better! Why? (Because we are no longer using a global context). So, in the second part of the swin transformer block, we displace the windows by $([M/2], [M/2])$ pixels from the regularly partitioned windows, and perform attention between these new windows! This leads to cross-window connections. In this case, since $M=4$, we displace the windows by $(2, 2)$.

