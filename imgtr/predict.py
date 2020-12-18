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

# from: 
# https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/

def softmax(ary):
    return np.exp(ary) / sum(np.exp(ary))

def load_weights(chkpt):
    with gzip.GzipFile(chkpt) as gz:
        obj = pickle.load(gz)
    return obj["flat_weights"]

def beam_search(data, k=3):
    import pprint
    from math import log
    sequences = [[list(), 0.0]]
    # walk over each step in sequence
    print(data.shape)
    for (r_num, row) in enumerate(data):
        print(r_num)
        #pprint.pprint(sequences)
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            softrow = softmax(row)
            for j in range(len(row)):
                candidate = [seq + [j], score - log(softrow[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[:k]
    sequences = np.array(sequences)
    return sequences

def load_model(chkpt, n_tokens=None, batch_size=None, max_length=None):
    print(f"Loading {chkpt} for inference")
    model = trax.models.TransformerLM(n_tokens, max_len=max_length, mode='train')
    example = np.zeros([batch_size, max_length]).astype(np.int32)
    signature = trax.shapes.signature(example)
    model.init_from_file(chkpt, weights_only=True, input_signature=signature)
    return model

def sample(inp, model, templist=[0, .01, .25, .5]):
    fill_val = tokens.special_token("<fill>")
    pad_val = tokens.special_token("<pad>")
    """
    cut = inp.shape[-1] // 2
    inp = inp[:, :-cut]
    #pad = np.zeros((inp.shape[0], cut)).astype(np.int32)
    pad = np.ones((inp.shape[0], cut)).astype(np.int32) * fill_val
    batch = np.concatenate((inp, pad), axis=-1)
    """
    #model = tl.Accelerate(model)
    output = model(inp)
    for temp in templist:
        if temp == 0:
            samples = jnp.argmax(output, axis=-1)
        else:
            samples = tl.logsoftmax_sample(output, temperature=temp)
        yield (temp, samples)

def auto_regressive_sample(inp, model, temperature=0.0):
    (batch_size, max_length) = inp.shape
    inp = inp[:, :max_length // 2]

    return trax.supervised.decoding.autoregressive_sample(
            model,
            inp,
            start_id=-1,
            eos_id=-1,
            batch_size=batch_size,
            temperature=temperature,
            max_length=max_length
    )

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

    inp = batch[0]

    model = load_model(checkpoint, n_tokens, batch_size, max_length)
    tok_images = sample(inp, model)
    #tok_images = auto_regressive_sample(inp, model)

    row = [tokens.tokens_to_image_array(ary, bitdepth=bitdepth) for ary in batch[0]]
    rows = [np.vstack(row)]
    for (temp, row) in tok_images:
        row = [tokens.tokens_to_image_array(it, bitdepth=bitdepth) for it in row]
        row = np.vstack(row)
        rows.append(row)
    row = [tokens.tokens_to_image_array(ary, bitdepth=bitdepth) for ary in batch[1]]
    row = np.vstack(row)
    rows.append(row)
    n_rows = len(rows)
    tbl = np.hstack(rows)
    img = tokens.array_to_image(tbl)
    width = scale * n_rows
    height = scale * batch_size
    img = img.resize((width, height))
    img.save("collage.png")

