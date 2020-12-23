# pixace

This is my pet project to experiment with modeling image data using [transformers](https://arxiv.org/abs/1706.03762). I only have access to a single 10GB GPU, so my objective was to train a network that balanced image complexity with quality.  If necessity is the mother of invention, design constraints are the midwives.

![Part of a complete breakfast](https://raw.githubusercontent.com/vishnubob/pixace/media/media/ttt-collage.jpg)

It uses decoder-only transformer and reformer architectures (language models) provided by [trax](https://github.com/google/trax).  In order to tackle the complexity of images, pixace reduces both the resolution and colorspace of each image before using it for training.  There a few design choices that went into this, but here is the terse version:

1. Load an image, scale it down
2. Convert to HSV colorspace
3. Quantize HSV channels with variable bit widths
4. Bitshift and pack the quantized HSV channels into a single integer
5. Flatten the image into a 1D array

Skipping the resize operation, here are a few examples of how this changes the color content of an image.  "HSV 544" means the image was quantized with five bits for hue (32 values), four bits for saturation (16 values), and four bits for shade (16 values), which is a total of 8192 unique colors.

| Original | HSV 544 | HSV 433 | HSV 322 |
| -------- | ------- | ------- | ------- |
| ![original](https://raw.githubusercontent.com/vishnubob/pixace/media/media/token_orig.jpg) | ![tokenized](https://raw.githubusercontent.com/vishnubob/pixace/media/media/token_5-4-4.jpg) | ![tokenized](https://raw.githubusercontent.com/vishnubob/pixace/media/media/token_4-3-3.jpg) | ![tokenized](https://raw.githubusercontent.com/vishnubob/pixace/media/media/token_3-2-2.jpg) |

For resizing, I decided to use 32x32 images.  These are tiny images, only 1024 pixels, but with interpolation, they scale up incredibly well.  After settling on HSV 544 for the colorspace and 32x32 image size, I was able to train a transformer model with batch size 8, or a reformer model with batch size 32 on a single 10gb card.  I used the [animal faces](https://www.kaggle.com/andrewmvd/animal-faces) dataset for my first attempt.

## Image Completion

Using [autoregressive sampling](https://trax-ml.readthedocs.io/en/latest/trax.supervised.html#trax.supervised.decoding.autoregressive_sample_stream), we can use our trained model to complete one half of an image:

![8x8 panel of generated images of animals](https://raw.githubusercontent.com/vishnubob/pixace/media/media/fill-in-example.jpg)

## Image Generation

With this same technique, we can also generate entirely new images:

![8x8 panel of generated images of animals](https://raw.githubusercontent.com/vishnubob/pixace/media/media/zoo-smol.jpg)

## How do I install it?

You will need to install this python package either as a local package or as a docker image.  There is a lot of package dependencies, so I recommend using docker.  The container is GPU enabled, but there is also a CPU only version of the dockerfile as well.  To use docker, just clone this repository and build the image:

```
# for GPU
git clone https://github.com/vishnubob/pixace/
cd pixace

# for GPU
docker build -t pixace .

# for CPU
docker build -f Dockerfile.cpu -t pixace .
```

If you don't have docker, you can use pip to install the package from github onto your local system:

```
sudo pip install https://github.com/vishnubob/pixace/archive/main.zip
```

This package installs a single command, also called `pixace`.  If you are running this from docker, the command is implicit in your invocation of the docker image.  Running it from docker looks something like:

```
docker run \
    -v $(pwd)/vol:/pixace \
    pixace \
    [command] [--arg1=...] [--arg2=...]
```

## Running pixace

Currently, pixace provides three top level commands: download, train, and predict.  

- `download` grabs compatible models from the internet (only one right now)
- `train` will train new models from scratch, or pickup training from a previous checkpoint.
- `predict` will create new images, either from scratch or with an image prompt.

Online help is available for any command by executing:

```
pixace COMMAND --helpfull
```

## Predictions using the animal faces reformer model

Before we can start to play with the animal faces model, we need to download it first.

```
# This will download the model weights to a directory called model-weights

pixace download --model_name=animalfaces
```

We can generate entirely new images from scratch:

```
# create four new images (no prompt) using the animal faces model
# save the result to predict.jpg

pixace predict \
    --model_name=animalfaces \
    --batch_size=4 \
    --out=predict.jpg
```

Or, we can generate new images based on a prompt image:

```
# Use image_1.jpg and image_2.jpg as image prompts
# Start the prediction at the 512th pixel of each prompt image
# Generate three different sampling temperatures (0.9, 1.0, 1.1)
# Save the result to prompt.jpg

pixace predict \
    --model_name=animalfaces \
    --prompt_image=image_1.jpg,image_2.jpg \
    --cut=512 \
    --temperature=0.9,1.0,1.1 \
    --out=prompt.jpg
```

## How to train your own pixace model

To start out, you will need to gather image data.  Currently, pixace is limited to modeling square images.  Feel free to use whatever aspect ratio you wish, but they will be squashed into squares regardless.  Plan on dedicating a fraction of your training set towards validation.  Validation sets are not strictly required, but if you do not provide one, pixace will use your training data for both training and validation.  Once your image data is curated, training a new model is as easy as:

```
pixace train \
    --model_name=my_model \
    --images=my_images/train \
    --val_images=my_images/val
```

Training will periodically update you on its metrics, but don't expect output right away.  It will automatically save training metrics to the weights directory, so you can also monitor your training session with tensorboard.  In addition to metrics, tensorboard will also visualize output from the model at each checkpoint.

There are a variety of configuration parameters such as `--batch_size`, `--bitdepth` and `--image_size`.  See the command line usage for more details.  For example, here is the usage for train:

```
pixace train --helpfull
<-- snip -->

pixace.flags:
  --batch_size: Batch size for training.
    (default: '16')
    (an integer)
  --bitdepth: HSV bitdepths
    (default: '5,4,4')
    (a comma separated list)
  --checkpoint: Path to checkpoint
  --image_size: Edge size for square image.
    (default: '32')
    (an integer)
  --images: Path to top level directory of images used for training
    (default: 'images/train')
  --model_dir: Top level directory for model data.
    (default: 'model-weights')
  --model_name: Model name.
    (default: '1223_0308')
  --n_epochs: Number of epochs
    (default: '10000')
    (an integer)
  --steps_per_epoch: Number of steps per epochs
    (default: '1000')
    (an integer)
  --val_images: Path to top level directory of images used for validation
    (default: 'images/val')

<-- snip -->
```
