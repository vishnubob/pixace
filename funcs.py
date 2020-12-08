import os
from PIL import Image
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from skimage.util import crop

class ImageBlocks(object):
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def to_blocks(self, img, strict=True):
        (w, h) = img.shape[:2]
        if strict:
            assert w % self.cols == 0
            assert h % self.rows == 0
        w_step = w // self.cols
        h_step = h // self.rows
        blocks = [
            img[col:col + w_step, row:row + h_step]
                for col in range(0, w, w_step)
                for row in range(0, h, h_step)
        ]
        return blocks

    def from_blocks(self, blocks):
        return np.block(blocks).reshape(self.cols, self.rows, -1)

    def per_block_mean(self, blocks):
        ch = blocks[0].shape[-1]
        N = blocks[0].reshape(ch, -1).shape[1]
        blocks = [np.sum(bl.reshape(ch, -1), axis=-1) / N for bl in blocks]
        return blocks

# hsv: 5, 4, 4
bitdepth = (5, 4, 4)

def quantize(img, bitdepth=bitdepth):
    maxvals = [2 ** bits - 1 for bits in bitdepth]
    img = np.round(img * maxvals).astype(np.int32)
    assert np.all(img <= maxvals)
    return img

def unquantize(img, bitdepth=bitdepth):
    maxvals = [2 ** bits - 1 for bits in bitdepth]
    img = img.astype(np.float) / maxvals
    assert np.all(img <= 1) and np.all(img >= 0)
    return img

def pack(img, bitdepth=bitdepth):
    img = quantize(img, bitdepth=bitdepth)
    img = img.T.reshape((img.shape[-1], -1))
    img[1] <<= bitdepth[0]
    img[2] <<= sum(bitdepth[:2])
    img = img[0] | img[1] | img[2]
    return img

def unpack(img, bitdepth=bitdepth):
    first = img & (2 ** bitdepth[0] - 1)
    second = (img >> bitdepth[0]) & (2 ** bitdepth[1] - 1)
    third = (img >> sum(bitdepth[:2])) & (2 ** bitdepth[2] - 1)
    img = np.vstack([first, second, third])
    img = unquantize(img.T, bitdepth=bitdepth)
    xy = int(img.shape[0] ** .5)
    assert xy ** 2 == img.shape[0]
    img = img.reshape((xy, xy, img.shape[-1]))
    return img

def image_to_array(img):
    img = np.array(img).astype(np.float) / 0xFF
    return img

def array_to_image(img_array, dnr=(0, 1)):
    if dnr:
        sf = dnr[1] - dnr[0]
        img_array = (img_array - dnr[0]) / sf
        img_array = img_array * 0xFF
    img_array = np.round(img_array).astype(np.uint8)
    # PIL requires greyscale to have only two dimensions
    if len(img_array.shape) == 3 and img_array.shape[-1] == 1:
        img_array = np.squeeze(img_array)
    o_img = Image.fromarray(img_array)
    return o_img

def process_image(img, blocker):
    img = img.resize((22, 22))
    img = image_to_array(img)
    #blocks = blocker.to_blocks(img)
    #blocks = blocker.per_block_mean(blocks)
    #img = blocker.from_blocks(blocks)
    img = rgb2hsv(img)
    img = quantize(img)
    #img = pack(img)
    return img

def process_array(img):
    #img = unpack(img)
    print(img.shape)
    img = unquantize(img)
    img = hsv2rgb(img)
    img = array_to_image(img)
    return img
