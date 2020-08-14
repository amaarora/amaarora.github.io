# EfficientNet Architecture 

1. TOC 
{:toc}

## Introduction
It brings me great pleasure as I begin writing about **EfficientNets** for three reasons:
1. At the time of writing, [Fixing the train-test resolution discrepancy: FixEfficientNet](https://arxiv.org/abs/2003.08237) (family of **EfficientNet**) is the current State of Art on ImageNet with **88.5%** top-1 accuracy and **98.7%** top-5 accuracy.
2. As far as I am aware, this is the only blog that explains the EfficientNet Architecture in detail **along with code implementations**.
3. This blog post also sets up the base for future blog posts on [Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/abs/1911.04252), [Fixing the train-test resolution discrepancy](https://arxiv.org/abs/1906.06423) and [Fixing the train-test resolution discrepancy: FixEfficientNet](https://arxiv.org/abs/2003.08237).

As an overview, not only will we look at the [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) research paper, but, we will also understand the nuances of the **EfficientNet** Architecture by looking at [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626) research paper. 

We will understand how the `EfficientNet-B0` architecture was first developed using **Nueral Architecture Search** similar to `MnasNet` approach and then scaled using **Compound Scaling** to get **EfficientNet B1-B7**. 

We also understand from a high level perspective how to implement the **EfficientNet** Architecture by looking [Ross Wightman](https://github.com/rwightman)'s wonderful implementation in [TIMM](https://github.com/rwightman/pytorch-image-models). This overview should provide the reader with sufficient code-level understanding to dig through the source code or carry out experiments of his/her own.

Currently, as far as I am aware, there are two main implementations of **EfficientNet** Architecure in PyTorch:
1. [Ross Wightman](https://github.com/rwightman)'s in [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
2. [Luke Melas](https://github.com/lukemelas)'s in [efficientnet_pytorch](https://github.com/lukemelas/EfficientNet-PyTorch)

Both are similar to the [official implementation](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) of **EfficientNet**s.

So, let's get started!

## The 'WHY'?
In this section we understand "why" **EfficientNet**s are totally worth your time. 

`fig-1` below summarizes "why" we could a learn a lot by understanding the **EfficientNet** Architecture.

![](/images/effnet_moment.png "fig-1 Model Size vs Imagenet Accuracy")

As we can see from `fig-1`, EfficientNets significantly outperform other ConvNets. In fact, `EfficientNet-B7` achieved new state of art with 84.4% top-1 accuracy outperforming the previous SOTA [GPipe](https://arxiv.org/abs/1811.06965) but being **8.4 times smaller** and **6.1 times faster**.

From the paper, 
> EfficientNet-B7 achieves state- of-the-art 84.4% top-1 / 97.1% top-5 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters.

The great thing about **EfficientNet**s is that not only do they have better accuracies compared to their counterparts, they are also lightweight and thus, faster to run. 

Having looked at their superior accuracies and faster runtimes, let's start to unravel the magic step-by-step.

## The "HOW"?
So "how" did the authors [Mingxing Tan](https://scholar.google.com/citations?user=6POeyBoAAAAJ&hl=en) and [Quoc V. Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ&hl=en) make **EfficientNet**s perform so well and efficiently?

In this section we will understand the main idea introduced in the research paper - **Compound Scaling**.

### Compound Scaling
Before the **EfficientNet**s came along, the most common way to scale up ConvNets was either by one of three dimensions - depth (number of layers), width (number of channels) or image resolution (image size). 

**EfficientNet**s on the other hand perform **Compound Scaling** - that is, scale all three dimensions while mantaining a balance between all dimensions of the network. 

From the paper:
> In this paper, we want to study and rethink the process of scaling up ConvNets. In particular, we investigate the central question: is there a principled method to scale up ConvNets that can achieve better accuracy and efficiency? Our empirical study shows that it is critical to balance all dimensions of network width/depth/resolution, and surprisingly such balance can be achieved by simply scaling each of them with constant ratio. Based on this observation, we propose a simple yet effective compound scaling method. Unlike conventional practice that arbitrary scales these factors, our method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients.

This main difference between the scaling methods has also been illustrated in `fig-2` below.

![](/images/model_scaling.png "fig-2 Model Scaling")

In `fig-2` above, (b)-(d) are conventional scaling that only increases one dimension of network width, depth, or resolution. (e) is the proposed compound scaling method that uniformly scales all three dimensions with a fixed ratio.

This main idea of **Compound Scaling** really set apart **EfficientNet**s from its predecessors. And intuitively, this idea of compound scaling makes sense too because if the input image is bigger (input resolution), then the network needs more layers (depth) and more channels (width) to capture more fine-grained patterns on the bigger image.

In fact, this idea of **Compound Scaling** also works on existing [MobileNet](https://arxiv.org/abs/1704.04861) and [ResNet](https://arxiv.org/abs/1512.03385) architectures. 
    
From `table-1` below, we can clearly see, that the scaled versions of **MobileNet** and **ResNet** architectures perform better than their baselines or also the conventional methods - (b)-(d) in `fig-2`. 

![](/images/effnet_t1.png "table-1 Compound Scaling")

Thus, it is safe to summarize - **Compound Scaling** works! 

### Nueral Architecture Search 
Since we are looking at the "how" - while so far we know **Compound Scaling** was the main idea introduced - the authors found that having a good baseline network is also critical. 

It wasn't enough to achieve such great performance by picking up any existing architecture and applying **Compound Scaling** to it. 

While the authors evaluated the scaling method using existing ConvNets, in order to better demonstrate the effectiveness of this scaling method, they also developed a new mobile-size baseline, called **EfficientNet** using **Nerual Architecture Search**. 

We understand this is in a lot more detail in this section of the blog post. 

### Summary
Therefore, to summarize the two main contributions of this research paper were the idea of **Compound Scaling** and using nueral architecture search to define a new mobile-size baseline called **EfficientNet**. We look at both model scaling and the **EfficientNet** architecture in a lot more detail in the following sections. 

## Model Scaling
Basically, the authors of **EfficientNet** architecture ran a few experiments scaling depth, width and image resolution and the two main observations that they made were: 

> 1. Scaling up any dimension of network width, depth, or resolution improves accuracy, but the accu- racy gain diminishes for bigger models.
> 2. In order to pursue better accuracy and efficiency, it is critical to balance all dimensions of network width, depth, and resolution during ConvNet scaling.

In this section, we will be looking at model scaling in a little more detail and get an understanding of how the authors came about those observations.

![](/images/scaling_effnet.png "fig-3 Scaling up a Baseline Model with Different Network Width(w), Depth(d) and Resolution(r)")

Particularly, in this section we will understand the results shown in `fig-3` and also get an understanding on why **Compound Scaling** works better than other traditional approaches. 

### Depth
Scaling network depth (number of layers), is the most common way used by many ConvNets. 

Thanks to Residual Connections and BatchNorm and other inventions, it has now been possible to train deeper nueral networks that generally have higher accuracy than their shallower counterparts. The intuition is that deeper ConvNet can capture richer and more complex features, and generalize well on new tasks. However, deeper networks are also more difficult to train due to the vanishing gradient problem. Also, residual connections and batchnorm help alleviate this problem, the accuracy gain of very deep networks diminishes. For example, ResNet-1000 has similar accuracy as ResNet-101 even though it has much more layers. 

In `fig-3` (middle), we can also see that ImageNet Top-1 Accuracy saturates at `d=6.0` and no further improvement can be seen after. 

### Width
Scaling network width - that is, increasing the number of channels in Convolution layers - is most commonly used for small size models. We have seen applications of wider networks previously in [MobileNets](https://arxiv.org/abs/1704.04861), [MNasNet]((https://arxiv.org/abs/1807.11626)). 

While wider networks tend to be able to capture more fine-grained features and are easier to train, extremely wide but shallow networks tend to have difficul- ties in capturing higher level features. 

Also, as can be seen in `fig-3` (left), accuracy quickly saturates when networks become much wider with larger `w`.

### Resolution
From the paper: 
> With higher resolution input images, ConvNets can potentially capture more fine-grained patterns. Starting from 224x224 in early ConvNets, modern ConvNets tend to use 299x299 (Szegedy et al., 2016) or 331x331 (Zoph et al., 2018) for better accuracy. Recently, GPipe (Huang et al., 2018) achieves state-of-the-art ImageNet accuracy with 480x480 resolution. Higher resolutions, such as 600x600, are also widely used in object detection ConvNets (He et al., 2017; Lin et al., 2017).

Increasing image resolution to help improve the accuracy of ConvNets is not new - This has been termed as Progressive Resizing in fast.ai course. (explained [here](https://www.kdnuggets.com/2019/05/boost-your-image-classification-model.html)).

It is also beneficial to ensemble models trained on different input resolution as explained by Chris Deotte [here](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/160147).

`fig-3` (right), we can see that accuracy increases with an increase in input image size. 

By studying the indivdiual effects of scaling depth, width and resolution, this brings us to the first observation which I re post here again for reference: 

> Scaling up any dimension of network width, depth, or resolution improves accuracy, but the accuracy gain diminishes for bigger models.

### Compound Scaling Experiments 
The authors also ran experiments to validation **Compound Scaling** idea. 

![](/images/compound_scaling.png "fig-4 Scaling Network Width for Different Baseline Net- works")

Each dot in a line in `fig-4` above denotes a model with different width(w). We can see that the best accuracy gains can be obvserved by increasing depth, resolution and width. `r=1.0` represents 224x224 resolution whereas `r=1.3` represents 299x299 resolution. 

Therefore, with deeper (d=2.0) and higher resolution (r=2.0), width scaling achieves much better accuracy under the same FLOPS cost.

This brings to the second observation: 

> In order to pursue better accuracy and efficiency, it is critical to balance all dimensions of network width, depth, and resolution during ConvNet scaling.

### Summary
So far, we have talked about the idea of **Compound Scaling** idea and looked at how this idea also works for existing ConvNets such as MobileNets and ResNet. 

We also looked at two main observations coming out of **Compound Scaling** experiments. 

Therefore, I think it is safe to summarize that best accuracy gains can be obvserved by increasing depth, resolution and width while mantaining balance between the three dimensions.

Having looked at **Compound Scaling**, we will now look the EfficientNet Architecture. 

## The EfficientNet Architecture

The authors used **Nueral Architecture Search** approach similar to **MNasNet**. This is a reinforcement learning based approach where the authors developed a baseline neural architecture **Efficient-B0** by leverage a multi-objective search that optimizes for **Accuracy** and **FLOPS**. From the paper: 

> Specifically, we use the same search space as (Tan et al., 2019), and use **ACC(m)×[FLOPS(m)/T]<sup>w</sup>** as the optimization goal, where `ACC(m)` and `FLOPS(m)` denote the accuracy and FLOPS of model `m`, `T` is the target FLOPS and `w=-0.07` is a hyperparameter for controlling the trade-off between accuracy and FLOPS. Unlike (Tan et al., 2019; Cai et al., 2019), here we optimize FLOPS rather than latency since we are not targeting any specific hardware device.

The EfficientNet-B0 architecture has been summarized in `table-2` below:

![](/images/effb0.png "Table-2 EfficientNet-B0 baseline network")

The `MBConv` layer above is nothing but an inverted bottleneck block with squeeze and excitation connection added to it. We will learn more about this layer in [this]() section of the blog post.

Starting from this baseline architecture, the authors scaled the **EfficientNet-B0** using **Compound Scaling** to obtain **EfficientNet B1-B7**. 

The overall approach can be easily summarized in the image below: 

![](/images/effnet_overall.jpg "fig-5 Overall Approach")

Basically, as summarized in `fig-5` above the overall approach of **EFficientNet**s involves using **Nueral Architecture Search** similar to the **MNasNet** approach to come up with **EfficientNet-B0** and then use **Compound Scaling** to scale this baseline network to get bigger, wider networks **EfficientNet B1-B7**.

### MnasNet Approach
Before we understand how was the **EfficientNet-B0** architecture developed, let's first look into the **MnasNet** Architecture and the main idea behind the research paper.

![](/images/Mnasnet.png "fig-6 An overview of MNasNet Approach")

From the paper:
> The search framework consists of three components: a recurrent neural network (RNN) based controller, a trainer to obtain the model accuracy, and a mobile phone based inference engine for measuring the latency.

For **MNasNet**, the authors used model accuracy (on ImageNet) and latency as model objectives to find the best architecture. 

Essentially, the Controller finds a model architecture, this model architecture is then used to train on ImageNet, it's accuracy and latency values are calculated. Then, reward function is calculated and feedback is sent back to controller. We repeat this process a few times until the optimum architecture is achieved such that it's accuracy is maximum given latency is lower than certain specified value. 

The objective function can formally be defined as: 

![](/images/mnasnet_obj.png)

Using the above as reward function, the authors were able to find the **MNasNet** architecture that achieved 75.2% top-1 accuracy and 78ms latency.

### Nueral Architecture Search for EfficientNets

The authors of the **EfficientNet** research paper used the similar approach as explained above to then find an optimal nueral network architecture that maximises **ACC(m)×[FLOPS(m)/T]<sup>w</sup>**. 
Note that for EfficientNets, the authors used FLOPS instead of latency in the objective function since the authors were not targeting specific hardware as opposed to **MNasNet** architecture.

From the paper: 
> Our search produces an efficient network, which we name EfficientNet-B0. Since we use the same search space as (Tan et al., 2019), the architecture is similar to MnasNett, except our EfficientNet-B0 is slightly bigger due to the larger FLOPS target (our FLOPS target is 400M). 

The authors named this architecture as **EfficientNet-B0** and it is defined in `table-2` shown below again for reference: 

![](/images/effb0.png "Table-2 EfficientNet-B0 baseline network")

Since, the authors of **EfficientNet**s used the same approach and similar nueral network search space as **MNasNet**, the two architectures are very similar. 

So, the key question now is - what's this **MBConv** layer? As I have mentioned before, it is nothing but an inverted residual bottleneck. 

This has been explained further in the next section. 

### Inverted Bottleneck `MBConv`

![](/images/mbconv.png "fig-7 MBConv Layer")

As in the case of Bottleneck layers that were used in the InceptionNet architecture and also in ResNet-50 and above, the key idea is to first use a `1x1` convolution to bring down the number of channels and apply the `3x3` or `5x5` convolution operation to the reduced number of channels to get output features. Finally, use another `1x1` convolution operation to again increase the number of channels to the initial value. 

The inverted bottleneck as in `MBConv` does the reverse - instead of reducing the number of channels, the first `1x1` conv layer increases the number of channels to 3 times the initial. 

Note that using a standard convolution operation here would be computationally expensive, so a Depthwise Convolution is used to get the output feature map. Finally, the second `1x1` conv layer downsamples the number of channels to the initial value. 

Now you might ask what's a Depthwise Convolution? It has been explained very well [here](https://www.youtube.com/watch?v=T7o3xvJLuHk).

So to summarize, the **EfficientNet-B0** architecture uses this inverted bottleneck with Depthwise Convolution operation. But, to this, they also add squeeze and excitation operation which have been explained in my previous blog post [here](https://amaarora.github.io/2020/07/24/SeNet.html).

