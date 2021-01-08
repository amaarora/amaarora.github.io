# The Annotated EfficientDet

First of all, a very happy new year to you! I really hope that 2021 turns out to be a lot better than 2020 for all of us. 

Today, we will be looking at the [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070) research paper. With single-model and single-scale, EfficientDet-D7 was able to achieve SOTA results at the time of release of the paper. Even after a year later, at the time of writing, the results are still in the top-5 positions on the [COCO leaderboard](https://paperswithcode.com/sota/object-detection-on-coco). Also, recently, in the [NFL 1st and Future - Impact Detection](https://www.kaggle.com/c/nfl-impact-detection/overview) Kaggle competition, **EfficientDets** figured in almost all of the [top winning solutions](https://www.kaggle.com/c/nfl-impact-detection/discussion/208812). So, in this blog post we will be uncovering all the magic that leads to such tremendous success for EfficientDets.

Also, while multiple blog posts previously exist on EfficientDets [[1](https://towardsdatascience.com/a-thorough-breakdown-of-efficientdet-for-object-detection-dc6a15788b73), [2](https://towardsdatascience.com/efficientdet-scalable-and-efficient-object-detection-review-4472ffc34fd9), [3](https://medium.com/towards-artificial-intelligence/efficientdet-when-object-detection-meets-scalability-and-efficiency-551e263719aa)..], there isn't one that explains how to implement `EfficientDets` in code. In another blog post that follows this one, I have explained how to implement EfficientDets in PyTorch. The implementation has been directly copied from [Ross Wightman's](https://twitter.com/wightmanr) excellent repo [efficientdet-pytorch](https://github.com/rwightman/efficientdet-pytorch).

So, let's get started. 

## Prerequisites
I assume that the reader has some knowledge about Object Detection. If you are completely new to the field or simply want to apply EfficientDet to a object detection problem, there are plenty examples on Kaggle that show how to use EfficientDets. This post is more meant for those who want to understand what's inside an EfficientDet.  

Also, here is a [great video](https://www.youtube.com/watch?v=8H2qGeOef44) of the fastai 2018 course by [Jeremy Howard](https://twitter.com/jeremyphoward) that introduces object detection. I started here too about a year ago. :)

Also, since it would be an overkill to put EfficientNets as a prerequisite, I will just say that it would be great if the reader has a good enough understanding of [EfficientNets](https://arxiv.org/abs/1905.11946). IfAlso you want a refresher, please refer to this [blog post](https://amaarora.github.io/2020/08/13/efficientnet.html) that explains EfficientNets in detail step-by-step. 

## Contributions 
There are two main contributions from the paper: 
1. A new version of a Feature Pyramid Network called **BiFPN**. 
2. And two, **Compund Scaling**. 

While the idea of compound scaling was first introduced in the EfficientNet paper, the authors apply it to object detection and achieve SOTA results. Also, that the authors of the EfficientDet research paper are the same authors who introduced EfficientNets. 

We will be looking at what the BiFPN network is in a lot more detail at a later stage.

## Introduction
This paper starts out with a similar introduction as EfficientNets where the authors explain why model efficiency becomes increasingly important for object detection. From the paper: 
> Tremendous progresses have been made in recent years towards more accurate object detection. meanwhile, state-of-the-art object detectors also become increasingly more expensive. For example, the latest AmoebaNet-based NASFPN detector requires 167M parameters and 3045B FLOPs (30x more than RetinaNet) to achieve state-ofthe-art accuracy. The large model sizes and expensive computation costs deter their deployment in many real-world applications such as robotics and self-driving cars where model size and latency are highly constrained. Given these real-world resource constraints, model efficiency becomes increasingly important for object detection.

The key question that this paper tries to solve is "**Is it possible to build a scalable detection architecture with both higher accuracy and better efficiency across a wide spectrum of resource constraints**"?

The authors identified two main challenges when it comes to answering this question: 
1. Efficient multi-scale feature fusion 
2. And two, Model Scaling 

To explain both, we will need to digress a little bit here and go back into the history of object detection when multi-scale feature fusion was first introduced by [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144). Let's look into what Feature Pyramid Networks are first. For compound scaling, refer [here](https://amaarora.github.io/2020/08/13/efficientnet.html#compound-scaling-1).

### Feature Pyramid Network for Object Detection
[He et al](https://arxiv.org/search/cs?query=He%2C+Kaiming&searchtype=author&abstracts=show&order=-announced_date_first&size=50) were the one of the first to exploit the inherent multi-scale pyramid heirarchy of CNNs and construct feature pyramids and apply to object detection. If this doesn't make sense right now - it's okay! I really havent explained in much detail what this means yet. That happens next. 

Recognizing objects at vastly different scales is a fundamental challenge in computer vision. Different authors have tried to solve this differently. However, there are four main categories of solutions that exist. 

![](/images/img_pyramid.png "fig-1 Different ways to recognize objects at different scales")

#### (a) Featurized Image Pyramid
This is the first way and possibly the simplest to understand to recognize objects at different scales. Given an input image, resize the image to using different scales, pass the original image and the resized images through a CNN, make a prediction at each scale and simply take the average of these predictions to get a final prediction. Intuitively, this enables a model to detect objects accross a large range of scales by scanning the model over both positions and pyramid levels. Chris Deotte explains this too in simple words [here](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/160147).

But, can you think of the possible problems with this possible solution? 

For one, inference time would increase. For each image, we would need to rescale it to various new sizes and then average the predictions. Second, it would also not be possible to do this during train time as this would be infeasible in terms of memory and hardware requirements. Therefore, featurized image pyramid technique can only be used at test time which creates an inconsistency between train/test time inference.  

#### (b) Single Feature Map (Faster RCNN)
Another way is to use the inherent scale invariance property of CNNs. As you know, during the forward pass, a deep CNN computes a feature heirarchy layer by layer, and therefore, has an inherent multi scale pyramidal shape. See VGG-16 network below as an example: 

![](/images/vgg16.png "fig-2 Inherent scale invariance in CNNs")

The later layers are much smaller in spatial dimensions compared to the earlier layers. Thus, the scales are different. 

Therefore, one could just accept an original image, do a forward pass through a CNN, and get bouding box and class predictions just using this single original scaled image making use of the inherent scale invariance property of CNNs. In fact, this is exactly what was done in the [Faster RCNN](https://arxiv.org/abs/1506.01497) research paper. 

![](/images/faster_rcnn.png "fig-3 Faster RCNN")

As can be seen in the image above, given an input image, we pass it through a CNN to get a 256-d long intermediate representation of the image. Finally, we use `cls layer` and `reg layer` to get classification and bounding box predictions in Faster RCNN method. This has been also explained very well by Jeremy in the video I referenced before.

Can you think of why this might not work? Maybe take a break to think about the possible reasons why this won't work.

As mentioned by Zeilur and Fergus in [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901) research paper, we know that the earlier layers of a CNN have low-level features whereas the deeper layers learn more and thus, have high-level features. The low-level features (understanding) in earlier layers of a CNN harm their representational capacity for object recognition. 

#### (c) Pyramidal Feature Heirarchy (SSD)
A third way could be to first have a stem (backbone) that extracts some meaning from the image and then have another convolutional network head on top to extract the features and perform predictions on each of the extracted features. This way, we do not need to worry about the representational capacity of earlier layers of a CNN. This sort of approach was introduced in the [SSD](https://arxiv.org/abs/1512.02325) research paper.

![](/images/ssd.png "fig-4 Single Shot Detector")

As can be seen in the image above, the authors of the research paper used earlier layers of VGG-16 (until `Conv5_3 layer`) to extract some meaning/representation of the image first. Then, they build another CNN on top of this and get predictions at each step or after each convolution. Infact, the SSD was one of the first attempts at using CNNs pyramidal feature heirarchy as if it were a featurized image pyramid. 

But can you think of ways to improve this? Well, to avoid using low-level features from earlier layers in a CNN, SSD instead builts the pyramid starting from high up in the network already (VGG-16) and then adds several new layers. But, while doing this, it misses the opportunity to reuse the earlier layers which are important for detecting small objects as shown in the FPN research paper. 

#### (d) Feature Pyramid Network
So, finally to the Feature Pyramid Network. Having had a look at all the other approaches, now we can appreciate what the FPN paper introduced and why it was such a success. In the FPN paper, a new architecture was introduced that combines the low-resolution, semantically strong features in the later layers with high-resolution, semantically weak features in the earlier layers via a top-down pathway and lateral connections. Thus, leading to **Multi-scale feature fusion**. 

The result is a feaure pyramid that has rich semantics at all levels because the lower semantic features are interconnected to the higher semantics. Somewhat similar idea to a [U-Net](https://amaarora.github.io/2020/09/13/unet.html). Also, since the predictions are generated from a single original image, the FPN network does not compromise on power, speed or memory.  
