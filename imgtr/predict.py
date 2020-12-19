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
from . globdb import GlobDatabase

def softmax(ary):
    return np.exp(ary) / sum(np.exp(ary))

def load_weights(chkpt):
    with gzip.GzipFile(chkpt) as gz:
        obj = pickle.load(gz)
    return obj["flat_weights"]

gin_config = \
""" 
trax.layers.SelfAttention.chunk_len = 64
trax.layers.SelfAttention.predict_drop_len = 128
trax.layers.SelfAttention.predict_mem_len = 1024
LSHSelfAttention.n_hashes = 4
"""

def load_gin():
    import gin
    gin.parse_config(gin_config)

def load_model(chkpt, n_tokens=None, batch_size=None, max_length=None):
    load_gin()
    print(f"Loading {chkpt} for inference")
    #model = trax.models.TransformerLM(n_tokens, max_len=max_length, mode='train')
    model = trax.models.ReformerLM(n_tokens, max_len=max_length, mode='predict')
    example = np.zeros([batch_size, max_length]).astype(np.int32)
    signature = trax.shapes.signature(example)
    model.init_from_file(chkpt, weights_only=True, input_signature=signature)
    return model

def sample_stream(batch, model, cut=None, temperature=1.0):
    #model = tl.Accelerate(model)
    fill_val = tokens.special_token("<fill>")
    pad_val = tokens.special_token("<pad>")
    (_, inp, _) = batch
    truth = inp[:, :]
    max_length = inp.shape[-1]
    cut = cut or max_length // 2

    for pos in range(cut, max_length - 1):
        inp = inp[:, :pos]
        pad = jnp.zeros((inp.shape[0], max_length - pos)).astype(jnp.int32)
        inp = np.concatenate((inp, pad), axis=-1)
        output = model(inp)
        samples = tl.logsoftmax_sample(output[:, :, :], temperature=temperature)
        inp = jax.ops.index_update(
            inp, 
            jax.ops.index[:, pos], 
            samples[:, pos]
        )
    yield inp
    yield truth

def autoreg(model, max_length):
    out = trax.supervised.decoding.autoregressive_sample(model, temperature=1.0, max_length=max_length)
    yield out

def sample(batch, model, cut=None, templist=[0, .1, .5, 1]):
    fill_val = tokens.special_token("<fill>")
    pad_val = tokens.special_token("<pad>")
    (inp, truth, _) = batch
    max_length = inp.shape[-1]
    if cut is not None:
        inp = inp[:, :-cut]
        pad = np.ones((inp.shape[0], max_length - cut)).astype(np.int32) * pad_val
        inp = np.concatenate((inp, pad), axis=-1)
        yield inp

    model = tl.Accelerate(model)
    output = model(inp)

    for temp in templist:
        if temp == 0:
            samples = jnp.argmax(output, axis=-1)
        else:
            samples = tl.logsoftmax_sample(output, temperature=temp)
        #samples = np.pad(samples, ((0, 0), (1, 0)))
        #samples = samples[:, :-1]
        yield samples
    yield truth

def predict_model(argv):
    output_dir = FLAGS.model_dir
    batch_size = FLAGS.batch_size
    bitdepth = FLAGS.bitdepth
    n_tokens = tokens.token_count(bitdepth=bitdepth)
    max_length = FLAGS.image_size ** 2
    checkpoint = FLAGS.checkpoint or f"{output_dir}/model.pkl.gz"
    scale = 256

    imgdb = GlobDatabase(FLAGS.images, "*.jpg")
    work_list = imgdb.select("val")
    itr = iter_dataset(work_list, batch_size=FLAGS.batch_size, group="")
    batch = next(itr)
    del itr

    model = load_model(checkpoint, n_tokens, batch_size, max_length)
    #tok_images = sample_stream(batch, model, cut=32 * 24)
    #tok_images = sample(batch, model, cut=max_length // 2)
    tok_images = autoreg(model, max_length)

    rows = []
    for row in tok_images:
        row = [tokens.tokens_to_image_array(it, bitdepth=bitdepth) for it in row]
        row = np.vstack(row)
        rows.append(row)

    n_rows = len(rows)
    tbl = np.hstack(rows)
    img = tokens.array_to_image(tbl)
    width = scale * n_rows
    height = scale * batch_size
    img = img.resize((width, height))
    img.save("collage.png")
