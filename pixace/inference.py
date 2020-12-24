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

from . import tokens
from . data import iter_dataset
from . flags import FLAGS
from . utils import download_image_from_web

def softmax(ary):
    return np.exp(ary) / sum(np.exp(ary))

def load_weights(chkpt):
    with gzip.GzipFile(chkpt) as gz:
        obj = pickle.load(gz)
    return obj["flat_weights"]

gin_config = \
""" 
trax.layers.SelfAttention.predict_drop_len = 128
trax.layers.SelfAttention.predict_mem_len = 1024
trax.layers.SelfAttention.chunk_len = 1024
LSHSelfAttention.n_hashes = 4
"""

def load_gin():
    import gin
    gin.parse_config(gin_config)

def load_model(chkpt, n_tokens=None, batch_size=None, max_length=None):
    load_gin()
    print(f"Loading {chkpt} for inference")
    # XXX: switch depending on gin
    #model = trax.models.TransformerLM(n_tokens, max_len=max_length, mode='train')
    model = trax.models.ReformerLM(n_tokens, max_len=max_length, mode='predict')
    example = np.zeros([batch_size, max_length]).astype(np.int32)
    signature = trax.shapes.signature(example)
    model.init_from_file(chkpt, weights_only=True, input_signature=signature)
    return model

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

def load_images(paths):
    size = FLAGS.image_size
    bitdepth = FLAGS.bitdepth
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

    toks = [tokens.image_to_tokens(img, size=size, bitdepth=bitdepth) for img in images] 
    toks = np.array(toks, dtype=np.int32)
    return toks

def build_collage(img_list, batch_size=None, scale=None):
    batch_size = batch_size or FLAGS.batch_size
    n_rows = len(img_list)
    bitdepth = FLAGS.bitdepth
    rows = []
    for row in img_list:
        row = [tokens.tokens_to_image_array(it, bitdepth=bitdepth) for it in row]
        row = [np.pad(it, ((1, 1), (1, 1), (0, 0))) for it in row]
        row = np.vstack(row)
        rows.append(row)
    tbl = np.hstack(rows)
    img = tokens.array_to_image(tbl)
    width = n_rows * scale
    height = batch_size * scale
    img = img.resize((width, height))
    return img

def predict_model(argv):
    bitdepth = FLAGS.bitdepth
    output_dir = FLAGS.model_dir
    batch_size = FLAGS.batch_size
    templist = list(map(float, FLAGS.temperature))
    n_tokens = tokens.token_count(bitdepth=bitdepth)
    max_length = FLAGS.image_size ** 2
    checkpoint = FLAGS.checkpoint or f"{output_dir}/model.pkl.gz"
    cut = FLAGS.cut
    scale = FLAGS.scale
    out = FLAGS.out

    out_images = []

    image_paths = FLAGS.prompt
    if image_paths:
        inp = load_images(image_paths)
        if inp.shape[0] > batch_size:
            inp = inp[:batch_size, :]
        elif inp.shape[0] < batch_size:
            batch_size = inp.shape[0]
    else:
        inp = None

    if inp is not None:
        if cut is None:
            cut = max_length // 2
        out_images.append(inp)
        inp = inp[:, :cut]
        pad = np.zeros((batch_size, max_length - cut), dtype=np.int32)
        actual = np.concatenate((inp, pad), axis=-1)
        out_images.append(actual)
    else:
        cut = 0

    length = max_length - cut - 1

    for temperature in templist:
        model = load_model(checkpoint, n_tokens, batch_size, max_length)
        row = autoreg(model, batch_size=batch_size, inp=inp, length=length, temperature=temperature)
        out_images.append(row)

    img = build_collage(out_images, batch_size=batch_size, scale=scale)
    img.save(out)
