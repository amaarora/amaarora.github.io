# The Annotated DETR

1. TOC 
{:toc}

## Introduction
Welcome to "**The Annotated DETR**". 

One of the most brilliant and well-explained articles I have ever read is [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html). It introduced **Attention** like no other post ever written. The simple idea was to present an "annotated" version of the paper [Attention is all you need](https://arxiv.org/abs/1706.03762) along with code.

Something I have always believed in is that when you write things in code, the implementation and secrets become clearer. Nothing is hidden anymore.

>  There is nothing magic about magic. The magician merely understands something simple which doesn’t appear to be simple or natural to the untrained audience. Once you learn how to hold a card while making your hand look empty, you only need practice before you, too, can “do magic.”
>
> -- Jeffrey Friedl in the book [Mastering Regular Expressions](https://learning.oreilly.com/library/view/mastering-regular-expressions/0596528124/ch01.html)

The **[DETR Architecture](https://arxiv.org/abs/2005.12872)** might seem like magic at first with all it's glitter and beauty too, but hopefully I would have uncovered that magic for you and revealed all the tricks by the time you finish reading this post. That is my goal. To make it as simple as possible for the keen readers to understand how the **DETR** model works underneath.

In this post, I am not trying to reinvent the wheel, but merely bringing together a list of prexisting excellent resources to make it easier for the reader to grasp DETR. I leave it up to the reader to further build upon these foundations in any area they choose.

> You can't build a great building on a weak foundation. You must have a solid foundation if you're going to have a strong superstructure. 
>
> -- Gordon B. Hinckley


## The DETR Architecture 
The overall DETR architecture is surprisingly simple and depicted in Figure-1 below. It contains three main components: a CNN backbone to extract a compact feature representation, an  encoder-decoder transformer, and a simple feed forward network (FFN) that makes the final detection prediction.

![](/images/detr_architecture.png "Figure-1: DETR Architecture")


### Backbone
Starting from the initial image $x_{img} ∈ R^3×H_0×W_0$ (with 3 color channels), a conventional CNN backbone generates a lower-resolution activation map $f ∈ R^{C×H×W}$. Typical values we use are C = 2048 and H,W = $\frac{H0}{32} , \frac{W0}{32}$.

```python 
class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
```

Above we create a simple backbone that inherits from `BackboneBase`. The backbone is created using `torchvision.models` and supports all models implemented in `torchvision`. For a complete list of supported models, refer [here](https://pytorch.org/vision/stable/models.html). 

As also mentioned above, the typical value for number of channels in the output feature map is 2048, therefore, for all models except `resnet18` & `resnet34`, the `num_channels` variable is set to 2048. This `Backbone` accepts a three channel input image tensor of shape $3×H_0×W_0$, where $H_0$ refers to the input image height, and $W_0$ refers to the input image width. 

Let's next look at the `BackboneBase` class that `Backbone` inherits from.

#### `BackboneBase`
```python 
class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out
```

The `BackboneBase` class above accepts a `tensor_list` which is a `NestedTensor`. `NestedTensor`s have been explained in the next section. But for now, it should suffice to know that they combine two tensors - `tensors` and `mask`.

**Side note**: Even though the `mask` argument is singular, from my understanding, this class accepts multiple masks for every tensor in the batch thus could very well be named `masks` instead of `mask`. 

The `forward` method of `BackboneBase` accepts an instance of `NestedTensor` class that we contains `tensors` and `mask`. `BackboneBase` then takes the `tensors` from `tensor_list` (instance of `NestedTensor`), and passes that through `self.body` which is responsible for getting the output feature map of shape $f ∈ R^{C×H×W}$, where $C$ is typically set to 2048. For an introduction to `IntermediateLayerGetter`, please refer to another blog post of mine - <enter blog link here>.

So, the output of `self.body` is a `Dict` that looks something like `{"0": <torch.Tensor>}` or `{"0": <torch.Tensor>, "1": <torch.Tensor>, "2": <torch.Tensor>...}` depending on whether `return_interm_layers` is `True` or `False`. Finally, we iterate through this `Dict` output of `self.body` which we call `xs`, interpolate the mask to have the same $H$ and $W$ as the lower-resolution activation map $f ∈ R^{C×H×W}$ output from `Backbone`. 

```python 
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
```

**Remember**: When we passed the `tensor_list.tensors` through `self.body` we updated the tensor which was first of size $R^3×H_0×W_0$ (with 3 color channels), to a lower-resolution activation map of size $f ∈ R^{C×H×W}$. Thus, we `interpolate` the mask accordingly. 

#### `NestedTensor`
`NestedTensor` is a simple tensor class that puts `tensors` and `masks` together as below: 

```python 
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
```

As can be seen from the `NestedTensor` source code, it combines `tensors` and `mask` and stores them as `self.tensors` and `self.mask` attributes. 

This `NestedTensor` class is really simple - it has two main methods:
1. `to`: casts both `tensors` and `mask` to `device` (typically `"cuda"`) and returns a new `NestedTensor` containing `cast_tensor` and `cast_mask`.
2. `decompose`: returns `tensors` and `mask` as a tuple, thus decomposing the "nested" tensor.
