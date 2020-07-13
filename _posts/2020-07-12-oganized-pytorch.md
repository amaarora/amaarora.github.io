# How to organize PyTorch code? Use PytorchLightning!
Have you tried [PytorchLightning](https://github.com/PyTorchLightning/pytorch-lightning) already? If so, then you know why it's so cool. If you haven't, hopefully by the time you finish reading this post, you will find it pretty cool (the word 'it' could refer to this blogpost or the wonderful [PytorchLightning](https://github.com/PyTorchLightning/pytorch-lightning) library - I leave this decision to the reader).

Note: From here on, we refer to **PytorchLightning** as **PL**, cause it's a long name to type and I left my favourite keyboard at work.

For a while now, I was jealous of Tensorflow solely because it's possible to use the same script and run it on CPU, GPU or TPU! For example, take this [notebook](https://www.kaggle.com/cdeotte/triple-stratified-kfold-with-tfrecords) from my one of my favourite kagglers and - at the time of writing this blogpost - a researcher at NVIDIA - [Chris Deotte](http://chrisdeotte.com/) and also 4x Grandmaster! 
Just by using an appropriate [strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy) in Tensorflow, it is possible to run the same experiments on your choice of hardware without changing anything else really. That is the same script could run in TPU, GPU or CPU. 

If you've already worked on multi-GPU machines or used [torch XLA](https://pytorch.org/xla/release/1.5/index.html) to run things on TPU using PyTorch, then you know my rant. Changing hardware choices in PyTorch is not as convenient when it comes to this. I love PyTorch - I do, but just this one thing would make me really frustrated. 

Welcome [PL](https://github.com/PyTorchLightning/pytorch-lightning)! I wish I tried this library sooner.

In this blogpost, we will be going through an introduction to PL and implement all the cool tricks like - Gradient Accumulation, 16-bit precision training, and also add TPU/multi-gpu support - all in a few lines of code. We use PL to work on [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/c/siim-isic-melanoma-classification) challenge on Kaggle - currently I am in the top 10% on the public leaderboard, and in a future blogpost I will share how to train a competitive Melonama classifier. In this post, our focus will be on introducing PL and we use the ISIC competition as an example.

The first part of this post, is mostly about getting the data, creating a dataset and dataloader and the interesting stuff about PL comes in section-3 [The Lightning Module](). If this stuff bores you, feel free to skip forward to the model implemented in PL.

1. TOC 
{:toc}

## What's ISIC Melanoma Classification challenge?
From the [description](https://www.kaggle.com/c/siim-isic-melanoma-classification) on Kaggle,
>  Skin cancer is the most prevalent type of cancer. Melanoma, specifically, is responsible for 75% of skin cancer deaths, despite being the least common skin cancer. The American Cancer Society estimates over 100,000 new melanoma cases will be diagnosed in 2020. It's also expected that almost 7,000 people will die from the disease. As with other cancers, early and accurate detection—potentially aided by data science—can make treatment more effective. Currently, dermatologists evaluate every one of a patient's moles to identify outlier lesions or “ugly ducklings” that are most likely to be melanoma.

In this competition, the participants are asked to build a Melonama classifier that classifies to identify melonama in images of skin lesions. Typical lesion images look like the ones below:

![](/images/ISIC.png "Example of Lesion Images")

In this blogpost, we will use PL to build a solution that can tell the malign melonama images apart from the rest. 

> A side note: Deep learning has come a far way. Compare this to 2012 where [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)  was trained on multiple e GTX 580 GPU which has only 3GB of memory. To train on 1.2 million examples of Imagenet, the authors had to split the model (with just 8 layers) to 2 GPUs. It took 5-6 days to train this network. Today, it's possible to train in a [few hours](https://www.fast.ai/2018/04/30/dawnbench-fastai/) or [even minutes](https://arxiv.org/abs/1709.05011#:~:text=We%20finish%20the%20100%2Depoch,2048%20KNLs%20without%20losing%20accuracy). For ISIC, each epoch for size 256x256 is around 2mins including validation on a P100 GPU. 

## Getting the data
You can download the 256x256 version of the Jpeg images [here]().

## Melonama Dataset
Getting our data ready for ingestion into the model is one of the basic things that we need to do for every project. 

```python
class MelonamaDataset:
    def __init__(self, image_paths, targets, augmentations=None):
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path))
        target = self.targets[idx]

        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        
        return image, torch.tensor(target, dtype=torch.long)
```
The above dataset is a pretty simple class that is instantiated by passing in a list of `image_paths`. To get an item, it reads an image using `Image` module from `PIL`, converts to `np.array` performs augmentations if any and returns `target` and `image`.

We can use `glob` to get `train_image_paths` and `val_image_paths` and create train and val datasets respectively.

```python 
# psuedo code
train_image_paths = glob.glob("<path_to_train_folder>")
val_image_paths = glob.glob("<path_to_val_folder>")

sz = 256 #go bigger for better AUC score but slower train time

train_aug = train_aug = albumentations.Compose([
    RandomCrop(sz,sz),
    ..., #your choice of augmentations
    albumentations.Normalize(always_apply=True), 
    ToTensorV2()
])

val_aug = albumentations.Compose([
    albumentations.CenterCrop(sz, sz),
    albumentations.Normalize(always_apply=True),
    ToTensorV2()
])

train_dataset = MelonamaDataset(train_image_paths, train_targets, train_aug)
val_dataset = MelonamaDataset(val_image_paths, val_targets, val_aug)
```

Once we have our `datasets` ready, we can now create our dataloaders and let's inspect the train images as a sanity check. 

```python 
# Dataloaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=4)
```

```python
# visualize images
import torchvision.utils as vutils

def matplotlib_imshow(img, one_channel=False):
    fig,ax = plt.subplots(figsize=(16,8))
    ax.imshow(img.permute(1,2,0).numpy())

images= next(iter(train_loader))[0][:16]
img_grid = torchvision.utils.make_grid(images, nrow=8, normalize=True)
matplotlib_imshow(img_grid)
```
![](/images/train_images.png "fig-1: Train images")

Now that our dataloaders are done, and looking good, we are ready for some lightning for our Melonama classifier! 

## Lightning Module
PL takes away much of the boilerplate code. By taking away the [Engineering Code](https://pytorch-lightning.readthedocs.io/en/latest/introduction_guide.html#engineering-code) and the [Non-essential code](https://pytorch-lightning.readthedocs.io/en/latest/introduction_guide.html#non-essential-code), it helps us focus on the [Research code](https://pytorch-lightning.readthedocs.io/en/latest/introduction_guide.html#research-code)!

The [Quick Start](https://pytorch-lightning.readthedocs.io/en/stable/new-project.html) and [Introduction Guide](https://pytorch-lightning.readthedocs.io/en/stable/introduction_guide.html) on PL's official documentation are great resources to start!


### Model and Training 
Our model in PL looks something like: 

```python
class Model(LightningModule):
    def __init__(self, arch='efficientnet-b0'):
        super().__init__()
        self.base = EfficientNet.from_pretrained(arch)
        self.base._fc = nn.Linear(self.base._fc.in_features, 1)

    def forward(self, x):
        return self.base(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-4)

    def step(self, batch):
        x, y  = batch
        y_hat = self(x)
        loss  = WeightedFocalLoss()(y_hat, y.view(-1,1).type_as(y_hat))
        return loss, y, y_hat.sigmoid()

    def training_step(self, batch, batch_nb):
        loss, y, y_hat = self.step(batch)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        loss, y, y_hat = self.step(batch)
        return {'loss': loss, 'y': y.detach(), 'y_hat': y_hat.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        auc = self.get_auc(outputs)
        print(f"Epoch {self.current_epoch} | AUC:{auc}")
        return {'loss': avg_loss}
    
    def get_auc(self, outputs):
        y = torch.cat([x['y'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        # shift tensors to cpu
        auc = roc_auc_score(y.cpu().numpy(), 
                            y_hat.cpu().numpy()) 
        return auc
```

We are using [WeightedFocalLoss](https://amaarora.github.io/2020/06/29/FocalLoss.html#how-to-implement-this-in-code) from my previous blogpose, because this is an imbalanced dataset with only around 1.77% positive classes. 

And that's really it. This is all it takes in PL to create, train and validate a deep learning model. 

Shifting from PyTorch to PL is super easy. It took me around a few hours to read up the introduction docs and reimplement the ISIC model in PL. In some ways, I was able to draw comparisons to the wonderful [fastai](https://arxiv.org/abs/2002.04688) library in the sense that both the libarries make our lives easier. 

I find PL code is much more organized and compact. Also, when sharing solutions, everybody knows exactly where to look! The training look is in the `training_step` method, validation loop is inside the `validation_step` and methods like `forward` remain unchanged.

To train this model, we simply create a [Trainer](https://pytorch-lightning.readthedocs.io/en/stable/trainer.html) and call `.fit()`.

```python
debug = False
gpus = torch.cuda.device_count()
trainer = Trainer(gpus=gpus, max_epochs=2, 
                  num_sanity_val_steps=1 if debug else 0)

trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

## outputs
>>  Epoch 0 | AUC:0.8667878706561116
    Epoch 1 | AUC:0.8867006574533746
```

And that's really it. This is all it takes to create a baseline model in PL.

Now to the cool tricks as I mentioned at the start of this post - Gradient Accumulation, 16-bit precision training and training on the TPU.

## Gradient Accumulation
So now that our baseline model is ready, let's add gradient accumulation!

```python
trainer = Trainer(gpus=1, max_epochs=2, 
                  num_sanity_val_steps=1 if debug else 0, 
                  accumulate_grad_batches=2)
```

It's as simple as adding a single parameter in PL!

A typical workflow in PyTorch would look like: 

```python
accumulate_grad_batches=2
optimizer.zero_grad()
for b_idx, data in enumerate(train_loader):
    loss = model(**data, args=args, weights=weights)
    loss.backward()
        if (b_idx + 1) % accumulate_grad_batches == 0:
                # take optimizer every `accumulate_grad_batches` number of times
                optimizer.step()
                optimizer.zero_grad()
```

PL nicely takes this boilerplate code away from us and provides easy access to researchers to implement gradient accumulation. It is very helpful to have larger batch sizes on a single GPU. To read more about it, refer to [this great article](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255) by [Hugging Face](https://huggingface.co/)!


## 16-bit precision training 
16 bit precision can cut the memory usage by half and also speed up training dramatically. [Here](https://arxiv.org/pdf/1905.12322.pdf) is a research paper which provides comprehensive analysis on 16-bit precision training. 

For a more gentler introduction refer to the fastai docs [here](http://dev.fast.ai/callback.fp16#A-little-bit-of-theory) which has some great resources and explains mixed precision very nicely.

To add 16-bit precision training, we first need to make sure that we PyTorch 1.6+. PyTorch only [recently added native support](https://analyticsindiamag.com/pytorch-mixed-precision-training/) for Mixed Precision Training.

To download the latest version of PyTorch simply run 
```bash
!pip install --pre torch==1.7.0.dev20200701+cu101 torchvision==0.8.0.dev20200701+cu101 -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html
```

After this, adding 16-bit training is as simple as:
```python
trainer = Trainer(gpus=1, max_epochs=2, 
                  num_sanity_val_steps=1 if debug else 0, 
                  accumulate_grad_batches=2, precision=16)
```

If you want to continue to use an older version of PyTorch, refer [here](https://pytorch-lightning.readthedocs.io/en/latest/apex.html#apex-16-bit).

In a typical workflow in PyTorch, we would be using `amp` fron NVIDIA to directly manipulate the training loop to support 16-bit precision training which can be very cumbersome and time consuming. With PyTorch now adding support for mixed precision and with PL, this is really easy to implement. 

## TPU Support
Finally, we are down to my last promise of adding TPU support and being able to run this script on TPUs!

Here's a [post by Google](https://cloud.google.com/blog/products/ai-machine-learning/what-makes-tpus-fine-tuned-for-deep-learning) introducing TPUs and here is an [excellent blogpost](https://medium.com/bigdatarepublic/cost-comparison-of-deep-learning-hardware-google-tpuv2-vs-nvidia-tesla-v100-3c63fe56c20f) comparing various pieces of hardware. TPUs are typically 5 times faster than a V100 and reduce training times significantly.

To use a TPU, switch to [Google Colab](https://colab.research.google.com/notebooks/basic_features_overview.ipynb) or [Kaggle](http://kaggle.com/) notebooks with free TPU availability. For more information on TPUs, watch [this video](https://www.youtube.com/watch?v=kPMpmcl_Pyw) by Google again.

To train your models on TPU on PL is again very simple, download the required libraries and add a parameter to the trainer. :)

```python
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev

trainer = Trainer(gpus=1, max_epochs=2, 
                  num_sanity_val_steps=1 if debug else 0, 
                  accumulate_grad_batches=2, precision=16, 
                  tpu_cores=8)
```
[Here](https://www.kaggle.com/abhishek/accelerator-power-hour-pytorch-tpu) is a notebook by Abhishek Thakur for ISIC using TPUs with pure PyTorch. If you compare, you'd realise how easy it is now with PL to train on TPUs.