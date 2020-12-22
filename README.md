# pixace

pixace is a pet project to experiment with modeling image data with [transformers](https://arxiv.org/abs/1706.03762).  Since I only have access to a single 10GB GPU, my objective was to train a network that balanced complexity with quality.  If necessity is the mother of invention, design constraints are the midwives.

![zoo](https://raw.githubusercontent.com/vishnubob/pixace/media/media/zoo-smol.jpg)

Currently, it uses standard decoder-only transformer and reformer architectures provided by [trax](https://github.com/google/trax).  In order to tackle the complexity of images, pixace reduces both the resolution and colorspace of each sample before using it for training.  There a few design choices that went into this process, but here is the terse version:

1. Load an image, scale it down
2. Convert to HSV colorspace
3. Quantize HSV channels with variable bit widths
4. Bitshift and pack the quantized HSV channels into a single integer
5. Flatten the image into a 1D array

Skipping the resize operation, here are a few examples of how this changes the color or an image.  "HSV 544" means the image was quantized with five bits for hue (32 values), four bits for saturation (16 values), and four bits for shade (16 values), which is a total of 8192 unique colors.

| Original | HSV 544 | HSV 433 | HSV 322 |
| -------- | ------- | ------- | ------- |
| ![original](https://raw.githubusercontent.com/vishnubob/pixace/media/media/token_orig.jpg) | ![tokenized](https://raw.githubusercontent.com/vishnubob/pixace/media/media/token_5-4-4.jpg) | ![tokenized](https://raw.githubusercontent.com/vishnubob/pixace/media/media/token_4-3-3.jpg) | ![tokenized](https://raw.githubusercontent.com/vishnubob/pixace/media/media/token_3-2-2.jpg) |
