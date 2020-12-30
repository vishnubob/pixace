import os
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

from . tokens import ImageTokenModel
from . utils import download_image_from_web

# XXX: hack to make inference go for reformer

gin_config = \
"""
#inference/trax.layers.SelfAttention.predict_drop_len = 128
#inference/trax.layers.SelfAttention.predict_mem_len = 1024
inference/trax.layers.SelfAttention.chunk_len = 1024
#inference/LSHSelfAttention.n_hashes = 4
"""

gin.parse_config(gin_config)

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

def build_collage(img_list, batch_size=None, scale=None, tokenizer=None):
    n_rows = len(img_list)
    rows = []
    for row in img_list:
        row = [tokenizer.decode(it) for it in row] 
        row = [np.pad(it, ((1, 1), (1, 1), (0, 0))) for it in row]
        row = np.vstack(row)
        rows.append(row)
    tbl = np.hstack(rows)
    img = tokenizer.array_to_image(tbl)
    width = n_rows * scale
    height = batch_size * scale
    img = img.resize((width, height))
    return img

class Inference(object):
    def __init__(self, model_name=None, model_type=None, weights_dir=None, checkpoint=None, image_size=None, bitdepth=None):
        assert model_name
        self.model_name = model_name
        # XXX: migrate most of these params to gin
        self.model_type = model_type
        self.weights_dir = weights_dir
        self.bitdepth = bitdepth
        self.image_size = image_size
        self.tokenizer = ImageTokenModel(image_size=self.image_size, bitdepth=bitdepth)
        # XXX: max-len
        self.max_length = self.image_size ** 2
        self.n_tokens = self.tokenizer.n_tokens
        if checkpoint is None:
            checkpoint = os.path.join(self.weights_dir, self.model_name, "model.pkl.gz")
        with gin.config_scope('inference'):
            self.model = self.load_model(checkpoint)

    def load_model(self, checkpoint=None):
        msg = f"Loading {self.model_type} model from '{checkpoint}'"
        print(msg)
        if self.model_type == "transformer":
            model = trax.models.TransformerLM(self.n_tokens, max_len=self.max_length, mode='predict')
        elif self.model_type == "reformer":
            model = trax.models.ReformerLM(self.n_tokens, max_len=self.max_length, mode='predict')
        else:
            msg = f"Unknown model type '{self.model_type}'"
            raise ValueError(msg)
        model.init_from_file(checkpoint, weights_only=True)
        return model

    def reset_model(self, batch_size=None):
        weights = self.model.weights
        example = np.zeros([batch_size, self.max_length]).astype(np.int32)
        signature = trax.shapes.signature(example)
        self.model.init(signature)
        self.model.weights = weights

    def predict(self, batch_size=None, prompts=None, cut=None, temperature=1, scale=256):
        batch_size = batch_size or 1
        out_images = []

        if prompts:
            inp = load_images(prompts, self.tokenizer)
            batch_size = inp.shape[0]

            if cut is None:
                cut = self.max_length // 2
            out_images.append(inp)
            inp = inp[:, :cut]
            pad = np.ones((batch_size, self.max_length - cut), dtype=np.int32) * self.tokenizer.offset
            actual = np.concatenate((inp, pad), axis=-1)
            out_images.append(actual)
        else:
            inp = None
            cut = 0

        length = self.max_length - cut - 1

        if isinstance(temperature, (int, float)):
            temperature = [temperature] 

        for temp in temperature:
            temp = float(temp)
            self.reset_model(batch_size=batch_size)
            row = autoreg(self.model, batch_size=batch_size, inp=inp, length=length, temperature=temp)
            out_images.append(row)

        img = build_collage(out_images, batch_size=batch_size, scale=scale, tokenizer=self.tokenizer)
        return img

    @classmethod
    def _absl_main(cls, argv):
        from . flags import FLAGS

        instance = cls(
            model_name=FLAGS.model_name,
            model_type=FLAGS.model_type,
            weights_dir=FLAGS.weights_dir,
            checkpoint=FLAGS.checkpoint,
            image_size=FLAGS.image_size,
            bitdepth=FLAGS.bitdepth
        )

        img = instance.predict(
            batch_size=FLAGS.batch_size,
            temperature=FLAGS.temperature,
            prompts=FLAGS.prompt,
            cut=FLAGS.cut,
            scale=FLAGS.scale
        )

        img.save(FLAGS.out)
