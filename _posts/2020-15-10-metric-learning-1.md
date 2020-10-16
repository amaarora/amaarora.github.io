# METRIC LEARNING 01: Introduction to Metric learning and Center Loss

This blog post is a first in a series of total 4 blog posts on **Metric learning**. When I first started to learn about metric learning, I couldn't find any blog posts that explained the concepts in detail along with code implementation. I wish to fill the gap through this series and in this series of 4 blog posts we will be looking at:

1. [Center Loss](https://kpzhang93.github.io/papers/eccv2016.pdf) 
2. [A-Softmax Loss (Sphereface Loss)](https://arxiv.org/abs/1704.08063) 
3. [CosFace Loss](https://arxiv.org/abs/1801.09414)
4. [ArcFace Loss](https://arxiv.org/abs/1801.07698)

We will understand the concepts in detail and also look at how to implement the above loss functions using PyTorch. 

In today's blog posts we will start with the basics of metric learning and also look at **Center Loss**. We look at what are closed-set and open-set problems and why metric learning is needed for open-set type problems. We also look at code-level implementations of Center Loss which is one variant of Softmax Loss that helps reduce intra-class variation. 

## What is metric learning and what are it's applications?
The most commonly used CNNs perform feature learning and label prediction, mapping the input data to deep features (the output of the last hidden layer), then to the predicted labels, as shown in Fig. 1 below.

![](/images/metric-learning-1.png "fig-1 Typical framework of training CNNs")

In generic object, scene or action recognition, the classes of the possible testing samples are within the training set, which is also referred to close-set identification. Therefore, the predicted labels dominate the performance and **softmax loss** is able to directly address the classification problems. In this way, the label prediction (the last fully connected layer) acts like a linear classifier and the deeply learned features are prone to be separable.

<u>For face recognition task, the deeply learned features need to be not only separable but also discriminative. And, the deep features learned using Softmax Loss are separable but not discriminative enough.</u> With metric learning, the aim is to learn discriminative features as shown in Fig 1. above. <u>Intuitively, minimizing the intra-class variations while keeping the features of different classes separable is the key.</u>

**What is Softmax Loss**?
When I first started reading papers related to metric learning losses, I was confused about what is Softmax Loss? From what I knew, Softmax is an activation function but not a loss function. Well, Softmax Activation followed by Cross Entropy Loss is referred to as Softmax Loss in the literature.

For applications such as facial recognition and landmark recognition, since the number of classes is so large that is ineffecient to treat them as standard classification type problems. In the recent [Google Landmark Recognition 2020](https://www.kaggle.com/c/landmark-recognition-2020) competition on Kaggle, the training set had 81,313 classes alone and using standard softmax loss and treating this as a standard classification leads to suboptimal results as the test set might have images that are outside the classes contained in training set. Such type of problems are referred to as open-set problems. Generally, we work with closed-set problems where the test images are part of the training set classes but in cases where the number of classes is so large such as in landark recognition or facial recognition, these are referred to as open-set problems and learning discrimative features is key for performing well in such type of challenges. 

As mentioned in the [Arcface paper](https://arxiv.org/abs/1801.07698), Softmax loss has some drawbacks:  
1. The size of the linear transformation matrix W ∈ R<sup>d×n</sup> increases linearly with the identities number n; 
2. The learned features are separable for the closed-set classification problem but not discriminative enough for the open-set recognition problem.

Therefore, several variants have been proposed to enhance the discriminative power of the softmax loss. And one such variant is the [Center Loss](https://kpzhang93.github.io/papers/eccv2016.pdf) which we will be looking at next. 

## Center Loss: A Discriminative Feature Learning Approach for Deep Face Recognition
In this paper, the authors introduced a new loss function, namely Center Loss, to efficiently
enhance the discriminative power of the deeply learned features in neural networks. Specifically, the idea is to learn a center (a vector with the same dimension as a feature) for deep features of each class.

From the paper, 
> In the course of training, we simultaneously update the center and minimize the distances between the deep features and their corresponding class centers. The CNNs are trained under the joint supervision of the softmax loss and center loss, with a hyper parameter to balance the two supervision signals. Intuitively, the softmax loss forces the deep features of different classes staying apart. The center loss efficiently pulls the deep features of the same class to their centers.

![](/images/center_loss.jpg "fig-2 Center Loss Explained")

To understand center loss, let's take the above image as an example. Let's assume we have 5 input images, 3 of `cat` and 2 of label `dog`. We pass it through a neural network to get 512 dimension vector outputs (or a n-dim vector). Next, we take average of the `cat` output vectors and `dog` output vectors to get `cat` and `dog` class centers. Now, as you might imagine, we will have to average the output vectors based on labels at every iteration to continue updating the class centers. The center loss proposed in this paper, makes sure that the output vectors of a label are very close to that label's class centers.

The main contributions of the paper were: 
- New loss function (called center loss) to minimize the intra-class distances of the deep features.
- Proposed loss function is very easy to implement in the CNNs.
- New state-of-the-art under the evaluation protocol of small training set on [Megaface Challenge](http://arxiv.org/abs/1505.02108).

