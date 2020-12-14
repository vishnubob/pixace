import os
from PIL import Image
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from skimage.util import crop
from . flags import FLAGS

SpecialSymbols = (
    '<fill>',
    '<pad>',
)

def token_count(bitdepth=None):
    bitdepth = bitdepth or FLAGS.bitdepth
    return (2 ** sum(bitdepth)) + len(SpecialSymbols)

def special_token(symbol, bitdepth=None):
    bitdepth = bitdepth or FLAGS.bitdepth
    idx = SpecialSymbols.index(symbol)
    return (2 ** sum(bitdepth)) + idx

def quantize(img, bitdepth=None):
    maxvals = [2 ** bits - 1 for bits in bitdepth]
    img = np.round(img * maxvals).astype(np.int32)
    assert np.all(img <= maxvals)
    return img

def unquantize(img, bitdepth=None):
    maxvals = [2 ** bits - 1 for bits in bitdepth]
    img = img.astype(np.float) / maxvals
    img = np.clip(img, 0, 1)
    assert np.all(img <= 1) and np.all(img >= 0)
    return img

def pack(img, bitdepth=None):
    img = quantize(img, bitdepth=bitdepth)
    img[:, :, 1] <<= bitdepth[0]
    img[:, :, 2] <<= sum(bitdepth[:2])
    img = img[:, :, 0] | img[:, :, 1] | img[:, :, 2]
    img = np.ravel(img)
    return img

def unpack(img, bitdepth=None):
    # lop off anything too big, and wrap back
    img = img % token_count(bitdepth=bitdepth)
    first = img & (2 ** bitdepth[0] - 1)
    second = (img >> bitdepth[0]) & (2 ** bitdepth[1] - 1)
    third = (img >> sum(bitdepth[:2])) & (2 ** bitdepth[2] - 1)
    img = np.vstack([first, second, third]).T
    img = unquantize(img, bitdepth=bitdepth)
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
    if (len(img_array.shape) == 3) and (img_array.shape[-1] == 1):
        img_array = np.squeeze(img_array)
    o_img = Image.fromarray(img_array)
    return o_img

def image_to_tokens(img, size=16, bitdepth=None):
    bitdepth = bitdepth or FLAGS.bitdepth
    img = img.resize((size, size))
    img = image_to_array(img)
    img = rgb2hsv(img)
    img = pack(img, bitdepth=bitdepth)
    return img

def tokens_to_image_array(img, bitdepth=None):
    bitdepth = bitdepth or FLAGS.bitdepth
    img = unpack(img, bitdepth=bitdepth)
    img = hsv2rgb(img)
    return img

def tokens_to_image(img, bitdepth=None):
    bitdepth = bitdepth or FLAGS.bitdepth
    i_ary = tokens_to_image_array(img, bitdepth=bitdepth)
    img = array_to_image(i_ary)
    return img

