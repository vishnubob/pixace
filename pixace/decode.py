import os
import pprint
import pickle
import gzip
import trax
import jax
import numpy as np
from jax import numpy as jnp
from trax import layers as tl
from absl import logging
from PIL import Image
import gin

from . utils import download_image_from_web

def adjust_reformer(scope="decode", chunk_len=None, n_hashes=4):
    assert chunk_len
    gin_path = f"{scope}/trax.layers.SelfAttention.chunk_len"
    gin.bind_parameter(gin_path, chunk_len)
    gin_path = f"{scope}/LSHSelfAttention.n_hashes"
    gin.bind_parameter(gin_path, n_hashes)

def autoreg(model, batch_size=1, inp=None, length=1, temperature=1.0):
    out = trax.supervised.decoding.autoregressive_sample_stream(
        model, 
        inputs=inp,
        batch_size=batch_size,
        temperature=temperature
    )

    result = []
    for sample in out:
        sample = sample[:, None]
        result.append(sample)
        if len(result) > length:
            break

    result = np.concatenate(result, axis=-1)

    if inp is not None:
        result = np.concatenate((inp, result), axis=-1)
    
    return result

def load_images(paths, tokenizer):
    images = []
    for path in paths:
        if not os.path.exists(path):
            if path.lower().startswith("http"):
                # maybe a URL?
                img = download_image_from_web(path)
            else:
                msg = f"'{path}' does not exist"
                raise ValueError(msg)
        else:
            img = Image.open(path)
        images.append(img)

    toks = [tokenizer.encode_image(img) for img in images] 
    toks = np.array(toks, dtype=np.int32)
    return toks

# XXX: support more than just single image
def decode_output(img_list, batch_size=None, scale=None, tokenizer=None, image_key="image"):
    n_rows = len(img_list)
    assert n_rows > 0
    rows = []
    labels = []
    decodes = []
    for row in img_list:
        parts = [tokenizer.decode(it) for it in row] 
        decodes.append(parts)
        row = [np.array(it[image_key], dtype=np.uint8) for it in parts]
        row = [np.pad(it, ((1, 1), (1, 1), (0, 0))) for it in row]
        row = np.vstack(row)
        rows.append(row)
    tbl = np.hstack(rows)
    img = Image.fromarray(tbl)
    width = n_rows * scale
    height = batch_size * scale
    img = img.resize((width, height))
    return (img, decodes)

class Decoder(object):
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    @property
    def max_len(self):
        return self.tokenizer.max_len
    
    @property
    def n_tokens(self):
        return self.tokenizer.n_tokens

    @classmethod
    def load(cls, tokenizer=None, **kw):
        model = load_model(**kw)
        return cls(model=model, tokenizer=tokenizer)

    def reset_model(self, batch_size=None):
        weights = self.model.weights
        example = np.zeros([batch_size, self.max_len]).astype(np.int32)
        signature = trax.shapes.signature(example)
        self.model.init(signature)
        self.model.weights = weights

    def predict(self, batch_size=1, prompts=None, cut=None, temperature=1, scale=256):
        outputs = {}
        if prompts:
            prompts = prompts[:batch_size]
            inp = [tokenizer.encode(pt) for pt in prompts]
            mincut = min([len(enc) for enc in inp])
            cut = mincut if cut is None else min(cut, mincut)
            inp = np.vstack(inp)
            inp = inp[:, :cut]
            pad = np.ones((batch_size, self.max_len - cut), dtype=np.int32) * self.tokenizer['pad']
            actual = np.concatenate((inp, pad), axis=-1)
            out_images.append(actual)
        else:
            inp = None
            cut = 0

        length = self.max_len - cut - 1

        out_images = []
        for temp in temperature:
            temp = float(temp)
            self.reset_model(batch_size=batch_size)
            row = autoreg(self.model, batch_size=batch_size, inp=inp, length=length, temperature=temp)
            out_images.append(row)

        return decode_output(out_images, batch_size=batch_size, scale=scale, tokenizer=self.tokenizer)

    @classmethod
    def _absl_main(cls, argv):
        from . flags import FLAGS
        from . factory import get_factory

        factory = get_factory()
        model = factory.load_model(checkpoint=FLAGS.checkpoint, mode="predict")
        decoder = cls(model=model, tokenizer=factory.tokenizer)

        if isinstance(FLAGS.temperature, (int, float)):
            FLAGS.temperature = [FLAGS.temperature] 

        (img, decodes) = decoder.predict(
            batch_size=FLAGS.batch_size,
            temperature=FLAGS.temperature,
            prompts=FLAGS.prompt,
            cut=FLAGS.cut,
            scale=FLAGS.scale
        )

        img.save(FLAGS.out)
        cols = [list() for x in range(len(FLAGS.temperature))]
        for row in decodes:
            for (idx, it) in enumerate(row):
                label = it["text"]
                cols[idx].append(label)
        for col in cols:
            print(col)
