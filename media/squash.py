#!/usr/bin/env python

import sys
sys.path.append("..")
from PIL import Image
from pixace import tokens

final = 256
large = 512
smol = 32

def write_example(bitdepth=None):
    ary = tokens.image_to_tokens(img, size=final, bitdepth=bitdepth)
    cvt = tokens.tokens_to_image(ary, bitdepth=bitdepth)
    cvt = cvt.resize((final, final))
    bitdepth_str = str.join('-', map(str, bitdepth))
    fn = f"token_{bitdepth_str}.jpg"
    cvt.save(fn)

img_fn = sys.argv[1]
img = Image.open(img_fn)
img = img.resize((final, final))
img.save("token_orig.jpg")

bitdepths = [
    (5, 4, 4),
    (4, 3, 3),
    (3, 2, 2),
    (2, 1, 1),
]

for bitdepth in bitdepths:
    write_example(bitdepth=bitdepth)
