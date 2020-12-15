import pickle
import gzip
import trax
import jax
from . beam_search import Search as BeamSearch

import numpy as np
from jax import numpy as jnp
from trax import layers as tl
from absl import logging

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

def beam_search(model, chkpt, inp, max_length=None, beam_size=1, temperature=0, alpha=0.0):
    weights = load_weights(chkpt)
    bs = BeamSearch(
        model,
        weights,
        max_length,
        beam_size=beam_size,
        temperature=temperature,
        alpha=alpha,
        eos_id=-1
    )
    inp = inp[:, :max_length // 2]
    return bs.decode(None, inp, batch_size=inp.shape[0])

def _beam_search(data, k=3):
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
    model = trax.models.TransformerLM(n_tokens, max_len=max_length, mode='predict')
    example = np.zeros([batch_size, max_length]).astype(np.int32)
    signature = trax.shapes.signature(example)
    model.init_from_file(chkpt, weights_only=True, input_signature=signature)
    return model

def greedy_search(batch, model, start_idx, max_length, temperature=0.5):
    model = tl.Accelerate(model)
    from tqdm import tqdm

    #for pos in range(start_idx, max_length):
    itr = tqdm(list(range(start_idx, max_length)))
    for pos in itr:
        output = model(batch)
        sample = tl.logsoftmax_sample(output[:, pos, :], temperature=temperature)
        #next_vals = jnp.squeeze(sample[:, -1])
        batch = jax.ops.index_update(
            batch, 
            jax.ops.index[:, pos], 
            sample
        )
    return batch

def predict_model(argv):
    output_dir = FLAGS.model_dir
    batch_size = FLAGS.batch_size
    bitdepth = FLAGS.bitdepth
    n_tokens = tokens.token_count(bitdepth=bitdepth)
    max_length = FLAGS.image_size ** 2
    chkpt = f"{output_dir}/model.pkl.gz"

    imgdb = GlobDatabase(FLAGS.images, "*.jpg")
    work_list = imgdb.select("train")
    itr = iter_dataset(work_list, batch_size=FLAGS.batch_size, group="train")
    #work_list = imgdb.select("val")
    #itr = iter_dataset(work_list, batch_size=FLAGS.batch_size, group="val")
    batch = next(itr)
    del itr

    inp = batch[0]
    orig = np.copy(inp)
    orig = [tokens.tokens_to_image(ary, bitdepth=bitdepth) for ary in orig]

    if 0:
        def t_cstr(mode='predict'):
            return trax.models.TransformerLM(n_tokens, max_len=max_length, mode=mode)

        tok_images = beam_search(t_cstr, chkpt, inp, max_length)

    if 1:
        inp = inp[:, :max_length // 2]
        pad_val = tokens.special_token("<pad>")
        pad = np.ones((inp.shape[0], max_length - max_length // 2)).astype(np.int32) * pad_val
        inp = np.concatenate((inp, pad), axis=-1)
        model = load_model(chkpt, n_tokens, batch_size, max_length)
        tok_images = greedy_search(inp, model, max_length // 2, max_length)

    if 0:
        model = load_model(chkpt, n_tokens, batch_size, max_length)
        start_id = -1
        inp = inp[:, :max_length // 2]

        #start_id = tokens.special_token("<fill>")
        #inp = inp[:, :max_length // 2 + 1]

        tok_images = trax.supervised.decoding.autoregressive_sample(
                model, 
                inp, 
                start_id=start_id,
                eos_id=-1,
                batch_size=batch_size,
                temperature=.25,
                max_length=max_length
        )

    for (idx, gen) in enumerate(tok_images):
        gen = tokens.tokens_to_image(gen, bitdepth=bitdepth)
        gen = gen.resize((512, 512))
        gen.save(f"example-{1 + idx:03d}.png")
        seed = orig[idx]
        seed = seed.resize((512, 512))
        seed.save(f"seed-{1 + idx:03d}.png")

