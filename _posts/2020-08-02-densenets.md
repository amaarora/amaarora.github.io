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

This is really it, as simple as this may sound, DenseNets essentially conect every layer to every other layer. This is the main idea that is extremely powerful.

From the paper: 
> DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters.

## But is feature concatenation possible? 
At this point in time, I want you to think about whether we can concat the features from the first layer of a **DenseNet** with the last layer of the **DenseNet**? If we can, why? If we can't, what do we need to do to make this possible? 

This is a good time to take a minute and think about this question. 

So, here's what I think - it would not be possible to concatenate the feature maps if the size of feature maps is different. So, to be able to perform the concatenation operation, we need to make sure that the size of the feature maps that we are concatenating is the same. Right?

But we can't just keep the feature maps the same size throughout the network - **an essential part of concvolutional networks is down-sampling layers that change the size of feature maps**. For example, look at the VGG architecture below: 

![](/images/imagenet_vgg16.png "fig-3 VGG architecture")

The input of shape *224x224x3* is downsampled to *7x7x512* towards the end of the network. 

To facilitate both down-sampling in the architecture and feature concatenation - the authors divided the network into multiple densely connected dense blocks. Inside the dense blocks, the feature map size remains the same.

![](/images/denseblock.png "fig-4 A DenseNet Architecture with 3 dense blocks")

Dividing the network into densely connected blocks solves the problem that we discussed above. 

Now, the `Convolution + Pooling` operations outside the dense blocks can perform the downsampling operation and inside the dense block we can make sure that the size of the feature maps is the same to be able to perform feature concatenation. 

### Transition Layers
The authors refer to the layers between the dense blocks as **transition layers** which do the convolution and pooling. 

The transition layers used in the **DenseNet** Architecutre from an implementation perspective consist of a batch-norm layer, 1x1 convolution followed by a 2x2 average pooling layer.

Given that the transition layers are prety easy, let's quickly implement them here:

```python
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
```

### Inside a single DenseBlock
Now that we understand that a DenseNet architecture is divided into multiple dense blocks, let's look at a dense block in a little more detail. Essentially, we know, that inside a dense block, each layer is connected to every other layer and the feature map size remains the same. 

![](/images/denseblock_single.jpeg "fig-5 A view inside the dense block")

Let's try and understand what's really going on inside a **dense block**. We have some gray input features that are then passed to `LAYER_0`. The `LAYER_0` performs a non-linear transformation to add purple features to the gray features. These are then used as input to `LAYER_1` which performs a non-linear transformation to also add orange features to the gray and purple ones. And so on until the final output for this 4 layer denseblock is a concatenation of gray, purple, orange and green features. 

As you can see the size of the feature map grows after a pass through each dense layer and the new features are concatenated to the existing features. One can think of the features as a global state of the network and each layer adds `K` features on top to the global state.

This parameter `K` is referred to as **growth rate** of the network.

## DenseNet Architecture as a collection of DenseBlocks

We already know by now from fig-4, that DenseNets are divided into multiple DenseBlocks.

The various architectures of DenseNet have been summarized in the paper.

![](/images/densenet_archs.png "fig-6 DenseNet Architectures")

Each architecture consists of four DenseBlocks with varying number of layers. For example, the `DenseNet-121` has `[6,12,24,16]` layers in the four dense blocks whereas `DenseNet-169` has `[6, 12, 32, 32]` layers.

We can see that the first part of the DenseNet architecture consists of a `7x7 stride 2 Conv Layer` followed by a `3x3 stride-2 MaxPooling layer`.

Also, the convolution operations inside each of the architectures are the Bottle Neck layers. What this means is that the `1x1 conv` reduces the number of channels in the input and `3x3 conv` performs the convolution operation on the transformed version of the input with reduced number of channels rather than the input.

### Bottleneck Layers
By now, we know that each layer produces `K` feature maps which are then concatenated to previous feature maps. Therefore, the number of inputs are quite high especially for later layers in the network. 

This has huge computational requirements and to make it more efficient, Bottleneck layers were introduced. From the paper:
> 1×1 convolution can be in- troduced as bottleneck layer before each 3×3 convolution to reduce the number of input feature-maps, and thus to improve computational efficiency. We find this design es- pecially effective for DenseNet and we refer to our network with such a bottleneck layer, i.e., to the BN-ReLU-Conv(1× 1)-BN-ReLU-Conv(3×3) version ofH?, as DenseNet-B. In our experiments, we let each 1×1 convolution produce 4k feature-maps.

We know `K` refers to the growth rate, so what the authors have finalized on is for `1x1 conv` to first produce `4*K` feature maps and then perform `3x3 conv` on these `4*k` size feature maps.

## DenseNet Implementation
We are now ready and have all the building blocks to implement DenseNet in PyTorch.

The first thing we need is to implement the dense layer inside a dense block.

```python
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        "Bottleneck function"
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features
```

A `DenseLayer` accepts an input, concatenates the input together and performs `bn_function` on these feature maps to get `bottleneck_output`. This is done for computational efficiency. Finally, the convolution operation is performed to get `new_features` which are of size `K` or `growth_rate`.

It should now be easy to map the above implementation with fig-5. Let's say the above is an implementation of `LAYER_2`. First, `LAYER_2` accepts the gray, purple and orange feature maps and concatenates them. 
Next, the `LAYER_2` performs a bottle neck operation to create `bottleneck_output` for computational efficiency. Finally, the layer performs the non linear transformation operation to generate `new_features`. These `new_features` are the green features as in fig-5.