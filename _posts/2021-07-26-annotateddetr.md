# The Annotated DETR

1. TOC 
{:toc}

## Foreword 
Welcome to "**The Annotated DETR**". 

One of the most brilliant and well-explained articles I have ever read is [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html). It introduced **Attention** like no other post ever written. The simple idea was to present an "annotated" version of the paper [Attention is all you need](https://arxiv.org/abs/1706.03762) along with code.

Something I have always believed in is that when you write things in code, the implementation and secrets become clearer. Nothing is hidden anymore.

>  There is nothing magic about magic. The magician merely understands something simple which doesn’t appear to be simple or natural to the untrained audience. Once you learn how to hold a card while making your hand look empty, you only need practice before you, too, can “do magic.”
>
> -- Jeffrey Friedl in the book [Mastering Regular Expressions](https://learning.oreilly.com/library/view/mastering-regular-expressions/0596528124/ch01.html)

The **[DETR Architecture](https://arxiv.org/abs/2005.12872)** might seem like magic at first with all it's glitter and beauty too, but hopefully I would have uncovered that magic for you and revealed all the tricks by the time you finish reading this post. That is my goal -

> To make it as simple as possible for the readers to understand how the **DETR** model works underneath.

In this post, I am not trying to reinvent the wheel, but merely bringing together a list of prexisting excellent resources to make it easier for the reader to grasp DETR. I leave it up to the reader to further build upon these foundations in any area they choose.

> You can't build a great building on a weak foundation. You must have a solid foundation if you're going to have a strong superstructure. 
>
> -- Gordon B. Hinckley

**NOTE:** All code referenced below has been copied from the [official DETR implementation](https://github.com/facebookresearch/detr).

## Introduction
We present a new method that views object detection as a direct set prediction problem. The main ingredients of the new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bi-partite matching, and a transformer encoder-decoder architecture. The new model is conceptually simple and does not require a specialized library, unlike many other modern detectors. DETR demonstrates accuracy and run-time performance on par with the well-established and highly-optimized Faster R-CNN baseline on the challenging COCO object detection dataset. Training code and pretrained models are available at <https://github.com/facebookresearch/detr>.

The goal of object detection is to predict a set of bounding boxes and category labels for each object of interest. Modern detectors address this set prediction task in an indirect way, by defining surrogate regression and classification problems on a large set of proposals, anchors, or window centers. Their performances are significantly influenced by postprocessing steps to collapse near-duplicate predictions, by the design of the anchor sets and by the heuristics that assign target boxes to anchors. To simplify these pipelines, we propose a direct set prediction approach to bypass the surrogate tasks. This end-to-end philosophy has led to significant advances in complex structured prediction tasks such as machine translation or speech recognition, but not yet in object detection: previous attempts either add other forms of prior knowledge, or have not proven to be competitive with strong baselines on challenging benchmarks. This paper aims to bridge this gap.

Our DEtection TRansformer (DETR, see Figure-1) predicts all objects at once, and is trained end-to-end with a set loss function which performs bipartite matching between predicted and ground-truth objects. DETR simplifies the
detection pipeline by dropping multiple hand-designed components that encode prior knowledge, like spatial anchors or non-maximal suppression. Unlike most existing detection methods, DETR doesn’t require any customized layers, and thus can be reproduced easily in any framework that contains standard CNN and transformer classes.

![](/images/DETR_overall.png "Figure-1: DETR directly predicts (in parallel) the final set of detections by combining a common CNN with a transformer architecture. During training, bipartite matching uniquely assigns predictions with ground truth boxes. Prediction with no match should yield a “no object” (∅) class prediction.")

DETR, however, obtains lower performances on small objects. Also, training settings for DETR differ from standard object detectors in multiple ways. 


## Data Preparation 
The input images are batched together, applying $0$-padding adequately to ensure they all have the same dimensions $(H_0,W_0)$ as the largest image of the batch.

> If you haven't worked with COCO before, the annotations are in a JSON format and must be converted to tensors before they can be fed to the model as labels. Refer to the [COCO website here](https://cocodataset.org/#format-data) for more information on data format.


### `CocoDetection` Dataset
The `CocoDetection` class below inherits from [torchvision's CocoDetection dataset](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.CocoDetection), and adds custom `_transforms` on top. There's also a custom `ConvertCocoPolysToMask` class that can prepare the dataset for both object detection and panoptic segmentation. 

```python 
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
```

Note that in `__getitem__`, we first use `torchvision.datasets.CocoDetection.__getitem__` to get `img, target`. The `img` here is returned as a `PIL.Image` instance and `target` is a list of `Dict`s for each annotation. Here, the `img, target` look like below: 

```python
img 
>> <PIL.Image.Image image mode=RGB size=640x427 at 0x7F841F918520>

target
>> [{'segmentation': [[573.81, 93.88, 630.42, 11.35, 637.14, 423.0, 569.01, 422.04, 568.05, 421.08, 569.97, 270.43, 560.38, 217.66, 567.09, 190.79, 576.69, 189.83, 567.09, 173.52, 561.34, 162.0, 570.93, 107.31, 572.85, 89.08]], 'area': 24373.2536, 'iscrowd': 0, 'image_id': 463309, 'bbox': [560.38, 11.35, 76.76, 411.65], 'category_id': 82, 'id': 331291}, {'segmentation': [[19.19, 206.3, 188.07, 204.38, 194.79, 249.48, 265.8, 260.04, 278.27, 420.28, 78.68, 421.24, 77.72, 311.85, 95.0, 297.46, 13.43, 267.71, 21.11, 212.06]], 'area': 42141.60884999999, 'iscrowd': 0, 'image_id': 463309, 'bbox': [13.43, 204.38, 264.84, 216.86], 'category_id': 79, 'id': 1122176}]
```

Next, some magic goes on when we do `img, target = self.prepare(img, target)`. After passing in the above `img` and `target` through `self.prepare`, the output looks like: 

```python
img
>> <PIL.Image.Image image mode=RGB size=640x427 at 0x7F841F918520>

target
>> {
    'boxes': tensor([[560.3800,  11.3500, 637.1400, 423.0000], [ 13.4300, 204.3800, 278.2700, 421.2400]]), 
    'labels': tensor([82, 79]), 
    'image_id': tensor([463309]), 
    'area': tensor([24373.2539, 42141.6094]), 
    'iscrowd': tensor([0, 0]), 
    'orig_size': tensor([427, 640]), 
    'size': tensor([427, 640])
}
```

In summary, the `prepare` method converted everything to a `tensor` and also instead of having a `List` of `Dict`s, the `target` is now a `Dict` with values as type `tensor`. Also, we are no longer returning segmentation masks since we are just working with Object Detection. There's some extra filtering that goes on inside `ConvertCocoPolysToMask` class like: 
1. Filter out objects if `iscrowd=1`.
2. Convert annotation from $[X, Y, W, H]$ to $[X_1, Y_1, X_2, Y_2]$ format. 
3. Filter out objects if $X_2 < X_1$ or $Y_2 < Y_1$.

I am going to skip over the exact source code of `ConvertCocoPolysToMask` but you can find it [here](https://github.com/facebookresearch/detr/blob/master/datasets/coco.py#L50-L112) if interested. 

Next, `self._transforms` are applied to `img, target`. The train transforms look like below: 

```python
normalize = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

if image_set == 'train':
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize(scales, max_size=1333),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600),
                T.RandomResize(scales, max_size=1333),
            ])
        ),
        normalize,
    ])
```

This is using [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html). From the paper: 

We use scale augmentation, resizing the input images such that the shortest side is at least 480 and at most 800 pixels while the longest at most 1333. To help learning global relationships through the self-attention of the encoder, we also apply random crop augmentations during training, improving the performance by approximately 1 AP. Specifically, a train image is cropped with probability 0.5 to a random rectangular patch which is then resized again to 800-1333.

This can be confirmed based on the train transforms above in code. Therefore, the final output from `CocoDetection` dataset is an `img` tensor and a `target` Dict.

### Custom collate function 
Next, the DETR architecture uses a custom collate function before the outputs from `CocoDetection` class are fed to the model. This is because each image is still of a different size and has not been converted to a `tensor` yet. As we already know, the input images are batched together, applying $0$-padding adequately to ensure they all have the same dimensions $(H_0,W_0)$ as the largest image of the batch.

This is how the collate function looks like: 

```python
def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)
```

All the fun occurs inside the `nested_tensor_from_tensor_list` function. But remember, here the batch is a list of length batch size. If batch size is 2, then, this `collate_fn` receives `batch` as a list of length 2, where `batch[0]` is the first item, that is a tuple containing `img` tensor, and `target` Dict (outputs from `CocoDetection` dataset).

By calling `batch = list(zip(*batch))`, we convert the batch of items into two lists, where the first list contains the images and the second list contains the targets. Finally, we pass the images to `nested_tensor_from_tensor_list` function that looks like: 


```python 
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)
```

This `nested_tensor_from_tensor_list` receives a `tensor_list`, which is a list of `img` tensors of varying shape. For example, the image shapes could be `[[3, 608, 911], [3, 765, 512]]` for a batch size of 2. Next, we get the `max_size` in `max_size = _max_by_axis([list(img.shape) for img in tensor_list])`. This `_max_by_axis` function returns the maximum value for each axis and looks like below:

```python 
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes
```

Therefore, the returned output is `[3, 765, 911]`. Finally, we define a `tensor` as `torch.zeros` of size `max_size` and also `mask` of size `max_size`. Next, we fill the tensor with `img` values and set `mask` to `False` inside the image shape values. Let me try and explain this visually: 

![](/images/input_img_detr.png "Figure-2: Input batch of 2 images of shapes - [[3, 765, 512], [3, 608, 911]].")

Now, both images get resized to the max size which is `[3, 765, 911]` and finally, `img` and `mask` values get filled as below. 

![](/images/collate_img_detr.png "Figure-3: Both images get resized to [3, 765, 911] and image and mask values get set in collate function.")

Here, the `blue` region and `orange` region in both respective resized images represent the filled values. For these `blue` and `orange` regions, the `mask` values are set to `False`, whereas in the grey region outside the `mask` values are set to `True`. Finally, these `tensor` and `mask` values are joined together in a `NestedTensor` class that has been explained later in this blog post. 

And that's it! We are now ready to feed the data to our DETR architecture. 

## The DETR Architecture 
The overall DETR architecture is surprisingly simple and depicted in Figure-1 below. It contains three main components: a CNN backbone to extract a compact feature representation, an  encoder-decoder transformer, and a simple feed forward network (FFN) that makes the final detection prediction.

![](/images/detr_architecture.png "Figure-2: DETR Architecture")


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

Since all the fun happens inside the `BackboneBase` class including the `forward` method, let's look at that next.

#### BackboneBase
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
The only difference between a standard Backbone like in [timm](https://github.com/rwightman/pytorch-image-models/) or any other framework and this `BackboneBase` class above is the input that the `BackboneBase` class expects. As can be seen in the `forward` method above, this `BackboneBase` expects the input to be of type `NestedTensor`. Now, what is this `NestedTensor`? We will look at that below. But, here its enough to know the key difference - any standard backbone in [timm](https://github.com/rwightman/pytorch-image-models/) expects a tensor as input, whereas, `BackboneBase` expects a `NestedTensor` as input. 

> A `NestedTensor` is basically 2 tensors - `tensors` and `mask` nested together. That's really all. More on `NestedTensor` in the next section. 

As already mentioned, the `forward` method of `BackboneBase` accepts an instance of `NestedTensor` class that contains `tensors` and `mask`. `BackboneBase` then takes the `tensors` from `tensor_list` (instance of `NestedTensor`), and passes that through `self.body` in `xs = self.body(tensor_list.tensors)`, which is responsible for getting the output feature map of shape $f ∈ R^{C×H×W}$, where $C$ is typically set to 2048. The `self.body` either returns the output from the last layer of the backbone model, or from all intermediate layers and the final layer depending on the value of `return_layers`. For an introduction to `IntermediateLayerGetter`, please refer to another blog post of mine - <enter blog link here>. 

The output of `self.body` is a `Dict` that looks something like `{"0": <torch.Tensor>}` or `{"0": <torch.Tensor>, "1": <torch.Tensor>, "2": <torch.Tensor>...}` depending on whether `return_interm_layers` is `True` or `False`. Finally, we iterate through this `Dict` output of `self.body` which we call `xs`, interpolate the mask to have the same $H$ and $W$ as the lower-resolution activation map $f ∈ R^{C×H×W}$ output from `Backbone`. Because, remember, the output from `self.body` will have a lower feature map size than the output. Because the backbone is a standard CNN, and it reduces the feature map size as we go deeper into the layers. Right? 

```python 
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
```

**Note**: Pointed this out again that when we passed the `tensor_list.tensors` through `self.body` we updated the tensor which was first of size $R^3×H_0×W_0$ (with 3 color channels), to a lower-resolution activation map of size $f ∈ R^{C×H×W}$. Thus, we `interpolate` the mask accordingly. 

**Note**: Please also note, that in summary, the Backbone is responsible for accepting an input `NestedTensor` that consists of the input image as `tensors` and a `mask` corresponding to the image. The backbone merely extracts the features from this input image, interpolates the mask to match the feature map size and returns them as a `NestedTensor` in a `Dict`. Okay? 

> Please re-read above notes until you understand them before moving forward. 

#### NestedTensor
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

Okay, great so far we now know how the `Backbone` has been implemented in the DETR architecture. But as can be seen from Figure-1, this is really a very small part of the overall architecture. There is a long way still to go.

#### Positional Encoding
> Going back to Figure-1, it can be seen that Positional Encodings are added to the output from the Backbone CNN.

Since the transformer architecture is permutation-invariant, we supplement it with [fixed positional encodings](https://arxiv.org/abs/1904.09925) that are added to the input of each attention layer. We defer to the supplementary material the detailed definition of the architecture, which follows the one described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

In simpler words, since all inputs are fed to the transformer parallely, instead of in a one-by-one fashion as in the case of RNNs, therefore, to let the transformer have information about the respective position of the image pixels, we add "positional encodings" to the input embeddings. 

```python 
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
```

As we already know from the [Attention is all you need](https://arxiv.org/abs/1706.03762) paper, the positional encodings can be mathematically defined as: 

$$ 
PE(pos, 2i) = \sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}}) \tag{1}
$$
$$ 
PE(pos, 2i) = \cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}}) \tag{2}
$$

where $pos$ is the position and $i$ is the dimension.

We defer the reader to [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) by [Amirhossein Kazemnejad](https://kazemnejad.com/about) for more information on Positional Encodings.

As for the `PositionalEncoding` class, remember, the `Backbone` first converts input image tensor of shape $3×H_0×W_0$  to a lower-resolution activation map of size $f ∈ R^{C×H×W}$. Positional Encodings are added to this lower-resolution feature map. Since, we need to be able to define positions both along the x-axis and y-axis, therefore, we have `y_embed` and `x_embed` variables that increase in value by 1 every time boolean `True` is present in `not_mask`. 

Next, we convert both `pos_x` and `pos_y` to be of dimension `dim_t` and finally take the alternate `.sin()` and `.cos()` to define `pos_x` and `pos_y`. In the end, `pos` becomes a concatenated tensor of `pos_x` and `pos_y`.

> This definition of Positional Encodings is very different from the ones I've seen before, you will find a simpler implementation of Positional Encodings in PyTorch Tutorial - [LANGUAGE MODELING WITH NN.TRANSFORMER AND TORCHTEXT](https://pytorch.org/tutorials/beginner/transformer_tutorial.html). But, please note, that those are 1-dimensional, whereas for `DETR`, we need 2-dimensional positional encodings for both the `X` and `Y` axis.

#### Joiner 
Since the transformer architecture is permutation-invariant, we supplement it with fixed positional encodings that are **added to the input of each attention layer**.

So far, we have only defined the positional encodings.  

```python
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos
```

There is really not much going on in the `Joiner` class. It accepts a `backbone` and `positional_embedding` (which is an instance of `PositionEmbeddingSine`) and initializes `nn.Sequential` with these two modules. 

Next, in the `forward` method, we pass the `tensor_list` through the backbone first to get lower-resolution activation map of size $f ∈ R^{C×H×W}$. Next, we store the outputs from the backbone in `out` and positional encodings in `pos` and return both lists. 

#### Summary 
So far, we have covered the covered the first section of the DETR architecture - Backbone and Positional Encodings. We still need to look at Transformer Encoder, Transformer Decoder and Attention Heads. Let's see how to implement the Transformer Architecture next.

![](/images/detr_architecture.png "Figure-1: DETR Architecture")

## Transformer Architecture
From Attention is all you need, the Transformer architecture has been presented in Figure-2 below: 

![](/images/Transformer-architecture.PNG "Figure-2: Transformer Architecture")

As can be seen, the Transformer Encoder, consists of multiple Transformer Encoder layers, where each layer consists of Multi-Head Attention and a feed-forward neural network. 

### Transformer Encoder
First, a 1x1 convolution reduces the channel dimension of the high-level activation map $f$ from $C$ to a smaller dimension $d$. creating a new feature map $z_0 ∈ R^{d×H×W}$. The encoder expects a sequence as input, hence we collapse the spatial dimensions of $z_0$ into one dimension, resulting in a $d×HW$ feature map. Each encoder layer has a standard architecture and consists of a **multi-head self-attention** module and a **feed forward network (FFN)**. 

In terms of implementation, from this point on the Transformer architecture is implemented very similarly to the implementation as explained in [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html). But, for completeness, I will also share the implementations below. 

The Transformer Encoder consists of multiple Transformer Encoder layers. Thus, it can be easily implemented as below: 

```python
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output
```

`_get_clones` simply clones the Transformer Encoder layer (explained below), `num_layers` number of times.  

**Note**: The Transformer Encoder, as can be seen from Figure-1, accepts the outputs from Backbone and also accepts Positional Encoding. Even though in Figure-1, it has been shown that the Positional Encoding and the output from the Backbone get added first, 

#### Transformer Encoder Layer

Below, the `TransformerEncoderLayer` has been implemented. 

```python 
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src
```

It's really straightforward, we accept an input which is called `src`, it is normalized using `nn.LayerNorm`, and finally we set the query and key matrices `q` and `k` as the same by adding positional encoding to the normalized input. Finally, self-attention operation is performed to get `src2` as the output. In this case, since we can attend to anywhere in the sequence - both forwards and backwards, `attn_mask` is actually set to None. Whereas, the `key_padding_mask` are the elements in the key that are ignored by attention. 

Next, is a simple feedforward layer with hidden dimension as `dim_feedforward`. And that's it! That's all the magic behind `TransformerEncoderLayer`. Next, let's look at the `TransformerDecoder`! 

### Transformer Decoder
The decoder follows the standard architecture of the transformer, transforming $N$ embeddings of size $d$ using multi-headed self-attention and encoder-decoder attention mechanisms. The difference with the original transformer is that our model decodes the $N$ objects in parallel at each decoder layer, while [Vaswani et al.]((https://arxiv.org/abs/1706.03762)) use an autoregressive model that predicts the output sequence one element at a time. We refer the reader unfamiliar with the concepts to the [supplementary material]((https://arxiv.org/abs/1706.03762)). Since the decoder is also permutation-invariant, the $N$ input embeddings must be different to produce different results. These input embeddings are learnt positional encodings that we refer to as object queries, and similarly to the encoder, we add them to the input of each attention layer. The N object queries are transformed into an output embedding by the decoder. They are then independently decoded into box coordinates and class labels by a feed forward network (described in the next subsection), resulting $N$ final predictions. Using self-attention and encoder-decoder attention over these embeddings, the model globally reasons about all objects together using pair-wise relations between them, while being able to use the whole image as context.

Similar to the Transformer Encoder, the Transformer Decoder consists of repeated Transformer Decoder layers and can be easily implemented as below:

```python
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)
```

The main difference is the option to return intermediate outputs for Auxilary losses (we'll look at losses later in the blog post). If `self.return_intermediate` is set to True, stacked output from every decoder layer is returned otherwise, output from the last decoder layer is returned. 

#### Transformer Decoder Layer
As for the Transformer Decoder layer, its implementation is also very similar to the Transformer Encoder layer. 

```python
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt
```