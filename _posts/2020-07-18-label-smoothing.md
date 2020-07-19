# Label Smoothing Explained using Microsoft Excel

In this blogpost, together, we: 
- Read and understand about Label Smoothing from [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) research paper
- Look at why do we need Label Smoothing?
- Reimplement Label Smoothing in Microsoft Excel step by step 
- Compare the results from our MS Excel implementation with `Fastai` and `PyTorch` versions of Label Smoothing 

**Why are we using Microsoft Excel?** 

It's a valid question you might ask and I wasn't a big fan of MS Excel either until I saw [this](https://youtu.be/CJKnDu2dxOE?t=7482) video by [Jeremy Howard](https://twitter.com/jeremyphoward) about [Cross Entropy Loss](https://en.wikipedia.org/wiki/Cross_entropy). In the video Jeremy explains Cross Entropy Loss using Microsoft Excel. It clicked and I understood it very well even with the fancy math in the cross entropy loss formula.

And that is my hope here too! In this blogpost I hope that together we can see past the mathematics and get the intuition for **Label Smoothing** and then later be able to implement it in a language/framework of our choice. 

So, let's get started! 

1. TOC 
{:toc}

## Why do we need Label Smoothing?
Let's consider we are faced with a multi-class image classification problem. Someone presents to us five images with labels -  

| Image Name | Label    |
|------------|----------|
| img-1.jpg  | Dog      |
| img-2.jpg  | Cat      |
| img-3.jpg  | Horse    |
| img-4.jpg  | Bear     |
| img-5.jpg  | Kangaroo |

As humans, we will quickly be able to assign labels to the image just by looking at them, for example we know that `img-1.jpg` is that of a dog, `img-2.jpg` is a cat and so on.

Let's one-hot encode the labels, so our labels get updated to:

| Image Name | is_dog | is_cat | is_horse | is_bear | is_kroo |
|------------|--------|--------|----------|---------|---------|
| img-1.jpg  | 1      | 0      | 0        | 0       | 0       |
| img-2.jpg  | 0      | 1      | 0        | 0       | 0       |
| img-3.jpg  | 0      | 0      | 1        | 0       | 0       |
| img-4.jpg  | 0      | 0      | 0        | 1       | 0       |
| img-5.jpg  | 0      | 0      | 0        | 0       | 1       |

Let's imagine that we used the above set of 5 images and the labels and trianed a deep learning model which in it's early stages learns to predict a set of LOGITS like so:

| Image Name | is_dog | is_cat | is_horse | is_bear | is_kroo |
|------------|--------|--------|----------|---------|---------|
| img-1.jpg  | 4.7      | -2.5      | 0.6        | 1.2       | 0.4       |
| img-2.jpg  | -1.2      | 2.4      | 2.6        | -0.6       |   2.34    |
| img-3.jpg  | -2.4      | 1.2      | 1.1        | 0.8       | 1.2       | 
| img-4.jpg  | 1.2      | 0.2      | 0.8        | 1.9       | -0.6       |
| img-5.jpg  | -0.9      | -0.1      | -0.2        | -0.5       | 1.6       |

This is pretty standard - this is what we do when we're training an image classifier anyway. Right? We pass a list of images and labels, make the model predict something, then calculate the cross-entropy loss and backpropogate to update the model's parameters. And we keep doing this until the model learns to assign the correct labels to the corresponding images. So what's the problem?

**Here's the important part:**

For the Cross-Entropy loss to really be at a minimum, each logit corresponding to the correct class needs to be **significantly higher** than the rest. That is, for example for row 1, `img-1.jpg` the logit of 4.7 corresponding to `is_dog` needs to be significantly higher than the rest. This is also the case for all the other rows where the logit corresponding to the true label needs to be significantly higher than the rest. 

A mathematical proof is presented [here](https://leimao.github.io/blog/Cross-Entropy-KL-Divergence-MLE/) by Lei Mao on why minimizing this loss is equivalent to maximising the logits. 

This case where, in order to minimise the cross-entropy loss, the logits corresponding to the true label need to be significantly higher than the rest can actually cause two problems.

From the paper, 
> This, however, can cause two problems. First, it may result in over-fitting: if the model learns to assign full probability to the ground- truth label for each training example, it is not guaranteed to generalize. Second, it encourages the differences between the largest logit and all others to become large, and this, combined with the bounded gradient `∂ℓ/∂z,k` , reduces the ability of the model to adapt.

In other words, the problem is that our model learns to become overconfident of it's predictions because to really minimise the loss, our model needs to be very sure of everything the model predicts. This is bad because it is then harder for the model to generalise and easier for it to overfit to the training data. We want the model to generalize and look be able to look at other dogs, cats.. images that weren't part of the training set and still be able to predict them well.

## What is Label Smoothing?

Label Smoothing was first introduced in [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567).  

From Section-7 - *Model Regularization via Label Smoothing* in the paper,
> We propose a mechanism for encouraging the model to be less confident. While this may not be desired if the goal is to maximize the log-likelihood of training labels, it does regularize the model and makes it more adaptable. The method is very simple. Consider a distribution over labels u(k), independent of the training example x, and a smoothing parameter Є. For a training example with ground-truth label y, we replace the label distribution q(k/x) = δ(k,y) with


![](/images/Label_Smoothing_Formula.png "eq-1")

> which is a mixture of the original ground-truth distribution q(k|x) and the fixed distribution u(k), with weights 1 − Є. and Є, respectively. 
> In our experiments, we used the uniform distribution u(k) = 1/K, so that

![](/images/label_smoothing_eq2.png "eq-2")

In other words, instead of using the hard labels or the one-hot encoded variables where the true label is 1, let's replace them with `(1-Є) * 1` where Є refers to the smoothing parameter. Once that's done, we add some uniform noise `1/K` where K: total number of labels.

So the updated distribution for the our examples with label smoothing factor `Є = 0.1` becomes:

| Image Name | is_dog | is_cat | is_horse | is_bear | is_kroo |
|------------|--------|--------|----------|---------|---------|
| img-1.jpg  | 0.92      | 0.02      | 0.02        | 0.02       | 0.02       |
| img-2.jpg  | 0.02      | 0.92      | 0.02        | 0.02       | 0.02       |
| img-3.jpg  | 0.02      | 0.02      | 0.92        | 0.02       | 0.02       |
| img-4.jpg  | 0.02      | 0.02      | 0.02        | 0.92       | 0.02       |
| img-5.jpg  | 0.02      | 0.02      | 0.02        | 0.02       | 0.92       |

We get the updated distribution above because `1-Є = 0.9`. So as a first step, we replace all the true labels with `0.9` instead of 1. Next, we add a uniform noise `1/K = 0.02` because in our case `K` equals 5. Finally we get the above update distribution with uniform noise.

The authors refer to the above change as *label-smoothing regularization* or LSR. And then we calculate the cross-entropy loss with the updated distribution LSR above instead.

Now we train the model with the updated LSR instead and therefore, cross-entropy loss get's updated to:

![](/images/LSR.png "eq-3")

Basically, the new loss `H(q′, p)` equals `1-Є` times the old loss `H(q, p)` + `Є` times the cross entropy loss of the noisy labels `H(u, p)`. This is key in understanding Label Smoothing - it is essentially the cross entropy loss with the noisy labels.

Let's now cut the math and implement this in Microsoft Excel step by step.

## Label Smoothing in Microsoft Excel
In this section we implement label smoothing in Microsoft Excel. We know that cross-entropy loss equals:

![](/images/cross_entropy.png "eq-4")

Great, and from section-2, we also know that Label Smoothing loss is actually the cross entropy loss with the noisy labels.

Let's consider we have five images again, but this time of only cats and dogs.

![](/images/ce_loss_ex1.png "fig-1")

Great, at the moment, the labels are one-hot encoded. Let's consider we are using a smoothing factor Є of 0.1. In this case, the updated labels become: 

![](/images/ce_loss_ex2.png "fig-2")

We get fig-2 by implementing eq-2 on fig-1. So, now we have our noisy labels. Next step is to simply calculate the cross-entropy loss. We will use the [fastai implementation of cross-entropy loss in excel](https://github.com/fastai/fastai/blob/master/courses/dl1/excel/entropy_example.xlsx), and use it on our noisy labels to calculate the Label Smoothing Cross Entropy Loss.

Let's consider that our model learns to predict the following LOGITS for each class like so: 

![](/images/ce_loss_ex3.png "fig-3")

Great, to calculate the loss, we first need to convert the LOGITS to probabilities. The LOGITS are the outputs from the last linear layer of our deep learning model. To convert them to probabilities, we generally have a softmax layer in the end. Jeremy explains how to implement Cross-Entropy loss [here](https://youtu.be/AcA8HAYh7IE?t=1844) including Softmax implementation. 

This is the where you PAUSE, look at the video and understand how Jeremy implements **Softmax** and **Cross-Entropy** loss in Microsoft Excel. If you already know how to implement Cross Entropy in Excel, great! 

Assuming, you have heard my advice, let's move on. We repeat the same process of applying **Softmax** to then get our probabilities from the LOGITS like so:

![](/images/ce_loss_ex4.png "fig-4")

So, now we have successfully converted LOGITS to Probabilities for each image. We did this by applying Softmax to the LOGITS.

The next step is simply to calculate the Cross-Entropy loss which is `∑q(x)log(p(x))` where p(x) refers to the predicted probability and `q(x)` refers to the ground truth label. In our case `q(x)` are the noisy labels, so we get the *LabelSmoothingCrossEntropy* loss like so:

![](/images/ce_loss_ex6.png "fig-5")

Believe it or not, we have just successfully implemented Label Smoothing cross entropy loss in Microsoft Excel.

## References 
1. [A Simple Guide to the Versions of the Inception Network](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202) by Bharat Raj
1. [When does label smoothing help](https://papers.nips.cc/paper/8717-when-does-label-smoothing-help.pdf) by Hinton et al
1. [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) aka Temperature Scaling by Pleiss et al 
1. [Mathematical explainations and proofs](https://leimao.github.io/blog/Label-Smoothing/) for label smoothing by Lei Mao
1. [Label Smoothing + Mixup](https://youtu.be/vnOpEwmtFJ8) by Jeremy Howard
1. [Cross Entropy Loss](https://youtu.be/CJKnDu2dxOE?t=7482) explained by Jeremy Howard

## Credits
This blogpost wouldn't have been possible without the help from my very talented friend - btw, who's also researching about [Instance Segmentation](https://paperswithcode.com/task/instance-segmentation/codeless) from Harvard - [Atmadeep Banerjee](https://twitter.com/abanerjee99). Atmadeep was very kind to jump on a call with me for over an hour and helped me find a mistake that I had missed - `LOG` function in excel has base 10 whereas in `numpy` and `pytorch` it's `LOG` to the base `e`! It was really funny to have spent the day reading numerous blogposts, few research papers and source code for PyTorch and then finding out that MS Excel implements `LOG` function differently than `numpy` and `pytorch`. But hey! Lesson learnt! When in doubt, contact [@Atmadeep Banerjee](https://twitter.com/abanerjee99)! He has an eye for detail.