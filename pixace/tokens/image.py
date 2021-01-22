import os
import numpy as np
from PIL import Image
from skimage.color import hsv2rgb, rgb2hsv
from . base import TokenModel
import imgaug.augmenters as iaa


class ImageTokenModel(TokenModel):
    def __init__(self, image_size=None, bitdepth=None, n_channels=3, colorspace='hsv', **kw):
        # XXX: support variable number of channels
        assert(len(bitdepth) == 3)
        if type(image_size) == int:
            image_size = (image_size, image_size)
        self.image_size = tuple(image_size)
        self.bitdepth = bitdepth
        self.n_channels = int(n_channels)
        self.colorspace = colorspace
        self.image_shape = self.image_size[::-1] + (self.n_channels,)
        max_len = self.image_size[0] * self.image_size[1]
        self.resizer = iaa.Sequential([
            iaa.CropToAspectRatio(self.image_size[0] / self.image_size[1]),
            iaa.Resize({"width": self.image_size[0], "height": self.image_size[1]}),
        ])

        super().__init__(max_len=max_len)

    @property
    def n_tokens(self):
        return super().n_tokens + (2 ** sum(self.bitdepth))

    def quantize(self, img):
        maxvals = [2 ** bits - 1 for bits in self.bitdepth]
        img = np.round(img * maxvals).astype(np.int32)
        assert np.all(img <= maxvals)
        return img

    def unquantize(self, img):
        maxvals = [2 ** bits - 1 for bits in self.bitdepth]
        img = img.astype(np.float) / maxvals
        img = np.clip(img, 0, 1)
        return img

    def _assert_shape(self, ary):
        assert ary.shape == self.image_shape, \
            f"input array shape {ary.shape} does not " \
            f"match expected image shape {self.image_shape}"
        
    def resize_image(self, img):
        return self.resizer(images=[img])[0]

    def encode(self, img): 
        self._assert_shape(img)
        img = self.quantize(img)
        img[..., 1] <<= self.bitdepth[0]
        img[..., 2] <<= sum(self.bitdepth[:2])
        img = img[..., 0] | img[..., 1] | img[..., 2]
        img = np.ravel(img)
        return super().encode(img)

    def decode(self, toks):
        toks = super().decode(toks)
        toks = self.pad_or_trim(toks, max_len=self.max_len - 2)
        first = toks & (2 ** self.bitdepth[0] - 1)
        second = (toks >> self.bitdepth[0]) & (2 ** self.bitdepth[1] - 1)
        third = (toks >> sum(self.bitdepth[:2])) & (2 ** self.bitdepth[2] - 1)
        img = np.vstack([first, second, third]).T
        img = self.unquantize(img)
        img = img.reshape(self.image_shape)
        return img

    def array_to_image(self, img):
        if self.colorspace == 'hsv':
            img = hsv2rgb(img)
        img = np.round(img * 0xFF).astype(np.uint8)
        # PIL requires greyscale to have only two dimensions
        if (len(img.shape) == 3) and (img.shape[-1] == 1):
            img = np.squeeze(img)
        img = Image.fromarray(img)
        return img

    def encode_image(self, img):
        if type(img) == str:
            img = Image.open(img)
        if img.mode != 'RGB':
            img = img.convert("RGBA").convert("RGB")
        img = np.array(img)
        img = self.resize_image(img)
        img = img.astype(np.float) / 0xFF
        if self.colorspace == 'hsv':
            img = rgb2hsv(img)
        return self.encode(img)

    def decode_image(self, toks):
        img = self.decode(toks)
        return self.array_to_image(img)
