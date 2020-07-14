# An introduction to PyTorch Lightning with comparisons to PyTorch 
Have you tried [PytorchLightning](https://github.com/PyTorchLightning/pytorch-lightning) already? If so, then you know why it's so cool. If you haven't, hopefully by the time you finish reading this post, you will find it pretty cool (the word 'it' could refer to this blogpost or the wonderful [PytorchLightning](https://github.com/PyTorchLightning/pytorch-lightning) library - I leave this decision to the reader).

Note: From here on, we refer to **PytorchLightning** as **PL**, cause it's a long name to type and I left my favourite keyboard at work.

For a while now, I was jealous of Tensorflow solely because it's possible to use the same script to train a model on CPU, GPU or TPU without really changing much! For example, take this [notebook](https://www.kaggle.com/cdeotte/triple-stratified-kfold-with-tfrecords) from my one of my favourite kagglers and - at the time of writing this blogpost - a researcher at NVIDIA - [Chris Deotte](http://chrisdeotte.com/) and also, since yesterday, Kaggle 4x Grandmaster! 
Just by using an appropriate [strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy) in Tensorflow, it is possible to run the same experiments on your choice of hardware without changing anything else really. That is the same script could run in TPU, GPU or CPU. 

If you've already worked on multi-GPU machines or used [torch XLA](https://pytorch.org/xla/release/1.5/index.html) to run things on TPU using PyTorch, then you know my rant. Changing hardware choices in PyTorch is not as convenient when it comes to this. I love PyTorch - I do, but just this one thing would make me really frustrated. 

Welcome [PL](https://github.com/PyTorchLightning/pytorch-lightning)! I wish I tried this library sooner.

In this blogpost, we will be going through an introduction to PL and implement all the cool tricks like - **Gradient Accumulation**, **16-bit precision training**, and also add **TPU/multi-gpu support** - all in a few lines of code. We use PL to work on [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/c/siim-isic-melanoma-classification) challenge on Kaggle. In this blogpost, our focus will be on introducing PL and we use the ISIC competition as an example.

We also draw comparisons to the typical workflows in PyTorch and compare how PL is different and the value it adds in a researcher's life.

The first part of this post, is mostly about getting the data, creating our train and validation datasets and dataloaders and the interesting stuff about PL comes in **The Lightning Module** section of this post. If this stuff bores you because you've done this so many times already, feel free to [skip](https://amaarora.github.io/2020/07/12/oganized-pytorch.html#lightning-module) forward to the model implemention.

1. TOC 
{:toc}

## What's ISIC Melanoma Classification challenge?
From the [description](https://www.kaggle.com/c/siim-isic-melanoma-classification) on Kaggle,
>  Skin cancer is the most prevalent type of cancer. Melanoma, specifically, is responsible for 75% of skin cancer deaths, despite being the least common skin cancer. The American Cancer Society estimates over 100,000 new melanoma cases will be diagnosed in 2020. It's also expected that almost 7,000 people will die from the disease. As with other cancers, early and accurate detection—potentially aided by data science—can make treatment more effective. Currently, dermatologists evaluate every one of a patient's moles to identify outlier lesions or “ugly ducklings” that are most likely to be melanoma.

In this competition, the participants are asked to build a Melonama classifier that classifies to identify melonama in images of skin lesions. Typical lesion images look like the ones below:

![](/images/ISIC.png "src: https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery")

In this blogpost, we will use PL to build a solution that can tell the malign melonama images apart from the rest. The model should take only a few hours to train and have 0.92 AUC score!

> A side note: Deep learning has come a far way. Compare this to 2012 where [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)  was trained on multiple e GTX 580 GPU which has only 3GB of memory. To train on 1.2 million examples of Imagenet, the authors had to split the model (with just 8 layers) to 2 GPUs. It took 5-6 days to train this network. Today, it's possible to train in a [few hours](https://www.fast.ai/2018/04/30/dawnbench-fastai/) or [even minutes](https://arxiv.org/abs/1709.05011#:~:text=We%20finish%20the%20100%2Depoch,2048%20KNLs%20without%20losing%20accuracy). For ISIC, each epoch for size 256x256 is around 2mins including validation on a P100 GPU. 

## Getting the data
You can download the 256x256 version of the Jpeg images [here](https://www.kaggle.com/cdeotte/jpeg-melanoma-256x256) with all the required metadata to follow along.

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
The above dataset is a pretty simple class that is instantiated by passing in a list of `image_paths`, `targets` and `augmentations` if any. To get an item, it reads an image using `Image` module from `PIL`, converts to `np.array` performs augmentations if any and returns `target` and `image`.

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
![](/images/train_images.png "Training images")

Now that our dataloaders are done, and looking good, we are ready for some lightning for our Melonama classifier! 

## Lightning Module
PL takes away much of the boilerplate code. By taking away the [Engineering Code](https://pytorch-lightning.readthedocs.io/en/latest/introduction_guide.html#engineering-code) and the [Non-essential code](https://pytorch-lightning.readthedocs.io/en/latest/introduction_guide.html#non-essential-code), it helps us focus on the [Research code](https://pytorch-lightning.readthedocs.io/en/latest/introduction_guide.html#research-code)!

The [Quick Start](https://pytorch-lightning.readthedocs.io/en/stable/new-project.html) and [Introduction Guide](https://pytorch-lightning.readthedocs.io/en/stable/introduction_guide.html) on PL's official documentation are great resources to start learning about PL! I started there too.


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

We are using [WeightedFocalLoss](https://amaarora.github.io/2020/06/29/FocalLoss.html#how-to-implement-this-in-code) from my previous blogpost, because this is an imbalanced dataset with only around 1.77% positive classes. 

### Model implementation compared to PyTorch
We add the `__init__` and `forward` method just like you would in pure PyTorch. The `LightningModule` just adds some extra functionalities on top. 

In pure pytorch, the `main` loop with training and validation would look something like:

```python
train_dataset, valid_dataset = MelonamaDataset(...), MelonamaDatasaet(...)
train_loader, valid_loader = DataLoader(train_dataset, ...), DataLoader(valid_dataset, ...)
optimizer = ...
scheduler = ...
train_augmentations = albumentations.Compose([...])
val_aug = albumentations.Compose([...])
early_stopping = EarlyStopping(...)
model = PyTorchModel(...)
train_loss = train_one_epoch(model, optimizer, scheduler)
preds, valid_loss = evaluate(args, valid_loader, model)
report_metrics()
if early_stopping.early_stop:
    save_model_checkpoint()
    stop_training()
```

And ofcourse, then we define our `train_one_epoch` and `evaluate` functions where the training loop looks typically like:
```python
model.train()
for b_idx, data in enumerate(train_loader):
    loss = model(**data)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```
And very similar for `evaluate`. As you can see, we have to write a lot of code to make things work in PyTorch. While this is great for flexibility, typically we have to reuse the same code over and over again in various projects. The training and evaluate loops hardly change much.

What PL does, is that it automates this process for us. No longer do we need to write the boilerplate code.

The training loop, goes directly inside the `training_step` method and the validation loop inside the `validation_step` method. The typical reporting of metrics happens inside the `validation_epoch_end` method. Inside the `Model` class, both the `training_step` and `validation_step` call the `step` method which get's the `x`s and `y`s from the batch, calls `forward` to make a forward pass and returns the loss. When we are finished training, our validation loop get's called and at the end of an epoch `validation_epoch_end` get's called which accumulates the results for us and calculates [AUC score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html). We use `roc_auc_score` because AUC score is used as a metric on the Kaggle competition itself.  

And that's really it. This is all it takes in PL to create, train and validate a deep learning model. There are some other nice functionalities like logging - `Wandb` and also `tensorboard` support which you can read more about [here](https://pytorch-lightning.readthedocs.io/en/latest/loggers.html).

Shifting from PyTorch to PL is super easy. It took me around a few hours to read up the introduction docs and reimplement the ISIC model in PL. I find PL code is much more organized and compact compared to PyTorch and still very flexible to run experiments. Also, when sharing solutions with others, everybody knows exactly where to look - for example, the training loop is always in the `training_step` method, validation loop is inside the `validation_step` and so on.

In some ways, I was able to draw comparisons to the wonderful [fastai](https://arxiv.org/abs/2002.04688) library in the sense that both the libraries make our lives easier. 

Similar to fastai, to train the model in PL, we can now simply create a [Trainer](https://pytorch-lightning.readthedocs.io/en/stable/trainer.html) and call `.fit()`.

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
[Here](https://www.kaggle.com/abhishek/accelerator-power-hour-pytorch-tpu) is a notebook by [Abhishek Thakur](https://www.linkedin.com/in/abhi1thakur/?originalSubdomain=no) for ISIC using TPUs with pure PyTorch. If you compare, you'd realise how easy it is now with PL to train on TPUs.


## Conclusion
So I hope by now, you were able to compare the differences between PyTorch and PL and that I have convinced you enough to at least try out PL. [Here] is an excellent Kaggle competition to practice those skills and use [PL](https://www.kaggle.com/c/tpu-getting-started)! In the first few experiments with PL, I have found my work to be more streamlined and also I have noticed a reduction in bugs. I find it easier to experiment with different batch sizes, mixed precision, loss functions, optimizers and also schedulers. PL is definitely worth a try.

## Credits
Thanks for reading! And please feel free to let me know via [twitter](https://twitter.com/amaarora) if you did end up trying PyTorch Lightning and the impact this has had on your experimentation workflows. Constructive feedback is always welcome.

- The implementation of Model was adapted and modified from [this](https://www.kaggle.com/hmendonca/melanoma-neat-pytorch-lightning-native-amp) wonderful notebook on Kaggle.