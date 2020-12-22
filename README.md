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

## How do I use it?

This is a work in progress, so please check back.  For the adventurous, there is a Dockerfile.  <3
