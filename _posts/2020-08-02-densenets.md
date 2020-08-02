# DenseNet Architecture Explained with PyTorch Implementation from TorchVision

1. TOC 
{:toc}

## Introduction 
In this post today, we will be looking at DenseNet architecture from the research paper [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993).

The overall agenda is to:
- Understand what DenseNet architecture is
- Introduce dense blocks, transition layers and look at a single dense block in more detail
- Understand step-by-step the TorchVision implementation of DenseNet

## DenseNet Architecture Introduction

In a standard Convolutional Neural Network, we have an input image, that is then passed through the network to get an output predicted label in a way where the forward pass is pretty straightforward as shown in the image below:

![](/images/CNN.png "fig-1 Convolutional Neural Network; src: https://cezannec.github.io/Convolutional_Neural_Networks/")

Each convolutional layer except the first one (which takes in the input), takes in the output of the previous convolutional layer and produces an output that is then passed to next convolutional layer. For `L` layers, there are `L` direct connections - one between each layer and its subsequent layer.  

The DenseNet architecture is all about modifying this standard CNN architecture like so:

![](/images/densenet.png "fig-2 DenseNet Architecture")

In a DenseNet architecture, each layer is connected to every other layer, hence the name **Densely Connected Convolutional Network**. For `L` layers, there are `L(L+1)/2` direct connections. For each layer, the feature maps of all the preceding layers are used as inputs, and its own feature maps are used as input for each subsequent layers.

From the paper: 
> DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters.

## But is feature concatenation possible? 
At this point in time, I want you to think about whether we can concat the features from the first layer of a **DenseNet** with the last layer of the **DenseNet**? If we can, why? If we can't, what do we need to do to make this possible? 

This is a good time to take a minute and think about this question. 

So, here's what I think - it would not be possible to concatenate the feature maps if the size of feature maps is different. So, to be able to perform the concatenation operation, we need to make sure that the size of the feature maps that we are concatenating is the same. Right?

But we can't just keep the feature maps the same size throughout the network - **an essential part of concvolutional networks is down-sampling layers that change the size of feature maps**. For example, look at the VGG architecture below: 

![](/images/imagenet_vgg16.png "fig-4 VGG architecture")

The input of shape *224x224x3* is downsampled to *7x7x512* towards the end of the network. 

To facilitate both down-sampling in the architecture and feature concatenation - the authors divided the network into multiple densely connected dense blocks. Inside the dense blocks, the feature map size remains the same.

## DenseNet Architecture as a combination of DenseBlocks

![](/images/denseblock.png "fig-3 A DenseNet Architecture with 3 dense blocks")

Dividing the network into densely connected blocks solves the problem that we discussed above. 

Now, the `Convolution + Pooling` operations outside the dense blocks can perform the downsampling operation and inside the dense block we can make sure that the size of the feature maps is the same to be able to perform feature concatenation. 

### Transition Layers
The authors refer to the layers between the dense blocks as **transition layers** which do the convolution and pooling. 

The transition layers used in the **DenseNet** Architecutre from an implementation perspective consist of a batch-norm layer, 1x1 convolution followed by a 2x2 average pooling layer.

### DenseBlock Explained
Now that we understand that a DenseNet architecture is divided into multiple dense blocks, let's look at a dense block in a little more detail. Essentially, we know, that inside a dense block, each layer is connected to every other layer and the feature map size remains the same. 

![](/images/denseblock_single.jpeg "fig-4 A view inside the dense block")

Let's try and understand what's really going on inside a **dense block**. We have some gray input features that are then passed to `LAYER_0`. The `LAYER_0` performs a non-linear transformation to add purple features to the gray features. These are then used as input to `LAYER_1` which performs a non-linear transformation to also add orange features to the gray and purple ones. And so on until the final output for this 4 layer denseblock is a concatenation of gray, purple, orange and green features. 

As you can see the size of the feature map grows after a pass through each dense layer and the new features are concatenated to the existing features. One can think of the features as a global state of the network and each layer adds `K` features on top to the global state.