# What is Focal Loss and when should you use it? 

In this blogpost we will understand what Focal Loss and when is it used. We will also take a dive into the math and implement it in PyTorch.

1. TOC 
{:toc}

## Q1: Where was Focal Loss introduced and what was it used for? 
Before understanding what Focal Loss is and all the details about it, let's first quickly get an intuitive understanding of what Focal Loss actually does. Focal loss was implemented in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) paper by He et al. 

For years before this paper, it was considered very hard to detect small size objects inside pictures during Object Detection. 

![](/images/IntroFL.PNG "Need for Focal Loss; [src: from fastai](https://youtu.be/0frKXR-2PBY?t=6223)")

What Focal Loss does is that it makes it easier for the model to predict things without being 80-100% sure that this object is "something". The reason, why in the image above, the bike is not predicted by the model, it's because Binary [Cross Entropy loss](https://en.wikipedia.org/wiki/Cross_entropy) (before Focal Loss) really asks the model to be confident about what is predicting whereas Focal Loss makes it easier for the model to predict things without being so sure. 

Focal Loss is particularly useful in cases where there is a class imbalance. For example, in the Object Detection image above, since most pixels are background and only very few pixels actually contain an object inside the image, this is a case of class imbalance. 

OK - so focal loss was introduced in 2017, and mostly used during class imbalance - great!

By the way, here are the predictions of the same model when trained with Focal Loss. 

![](/images/FL_preds.PNG "After Focal Loss training; [src: from fastai](https://youtu.be/0frKXR-2PBY?t=6223)")

See the difference? This time the model is predicting something for the bike! :) 

## Q2: So, why did that work? What did Focal Loss do to make it work?
So now that we have seen an example of what Focal Loss can do, let's try and understand why that worked. The most important bit to understand about Focal Loss is the graph below: 

![](/images/FL_v_CE.PNG "Compare FL with CE")

In the graph above, the "blue" line represents the Cross Entropy Loss. The X-axis or 'probability of ground truth class' (let's call it `Pt` for simplicity) is the probability that the model predicts for the object. 
As an example, let's say the model predicts that something is a bike with probability 0.6 and it actually is a bike. The in this case `Pt` is 0.6. 
Also, consider the same example but this time the object is not a bike. Then `Pt` is 0.4 because ground truth here is 0 and probability that the object is not a bike is 0.4 (1-0.6).

The Y-axis is simply the loss value given `Pt`. 

As can be seen from the image, when the model predicts the ground truth with a probability of 0.6, the Cross Entropy Loss is still somewhere around 0.5. Imagine, in the case of imbalance, if the model predicts rare class with probability 0.6, then the CE loss would add up significantly and therefore, in order to reduce Cross Entropy Loss, our model would have to be very confident about the class it predicts. 

Focal Loss is different that way. It reduces the loss for "well-classified examples" or examples when the model predicts the right thing with probability > 0.5 whereas, it increases loss for "hard-to-classify examples" when the model predicts with probability < 0.5.

The major difference is that losses from the rare class don't add up anymore in case of Focal Loss and therefore, the model doesn't need to be 80-100% sure about the thing it predicts to actually reduce loss. It could just as well be 40-60% sure and still the loss would be low. Therefore, in the second case of object detection, the model at least attempts to predict and have a go at the motorbike.


## Q3: Alpha and Gamma?
Now that we know what Focal Loss is doing, let's quickly get in to the math. 

![](/images/CE1.PNG "Cross Entropy Loss")

Cross Entropy Loss is negative log likelihood. [Here's]9https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a) an excellent blogpost that explains Cross Entropy Loss.

Therefore, `Pt` or 'probability of ground truth' can be defined as: 
![](/images/pt.PNG "Probability of Ground Truth")

So after, refactoring: 
![](/images/CE.PNG "Cross Entropy Loss")

From the Paper, 
> A common method for addressing class imbalance is to introduce a weighting factor α ∈ [0, 1] for class 1 and 1−α for class −1.

Therefore, `alpha` are the weights assigned to each class. And so, Weighted CE becomes: 

![](/images/Alpga_CE.PNG "Weighted Cross Entropy Loss")

While 'Weighted Cross Entropy Loss' is one way to deal with class imbalance, the paper reports: 
> The large class imbalance encountered during training of dense detectors overwhelms the cross entropy loss. Easily classified negatives comprise the majority of the loss and dominate the gradient. While α balances the importance of positive/negative examples, it does not differentiate between easy/hard examples.

This is the point where I said that if the model predicts something right with a probability of 0.6, the Cross Entropy Loss is still around 0.5 and all these losses accumulate and overpower the rare class. All alpha is doing, is adding some weights, but in cases where Rare class ratio is so low, even these weights are unable to help much. 

From the paper for Focal Loss: 
> We propose to add a modulating factor (1 − pt)**γ to the cross entropy loss, with tunable focusing parameter γ ≥ 0.

So, all the authors really have done is add `(1 − pt)**γ` to Cross Entropy Loss such that the Final Loss becomes: 

![](/images/FL_no_weight.PNG "Non Weighted Focal Loss")

Now the very first graph will make much more sense. Because for `pt` > 0.5, `1-pt` is less than 0.5 and adding a power of `γ` to it reduces the loss even further when compared to Cross Entropy Loss.

Finally, the Alpha weighted version of Focal Loss is:

![](/images/FL.PNG "Non Weighted Focal Loss")

## Q4: How to implement it in PyTorch? 

While TensorFlow provides this loss function [here](https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy), this is not inherently supported by PyTorch so we have to write a custom loss function. 

Here is the implementation of Focal Loss in PyTorch: 

```python 
class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
```