import trax
import jax

import numpy as np
from jax import numpy as jnp
from trax import layers as tl
from absl import logging

from . import tokens
from . data import iter_dataset
from . flags import FLAGS
from . globdb import GlobDatabase

def greedy_search(batch, model, start_idx, max_length):
    for pos in range(start_idx, max_length):
        new_batch = model(batch)
        next_vals = jnp.argmax(new_batch, axis=-1)
        next_vals = jnp.squeeze(next_vals[:, pos:pos+1])
        batch = jax.ops.index_update(
            batch, 
            jax.ops.index[:, pos], 
            next_vals
        )
    return batch

def predict_model(argv):
    output_dir = FLAGS.model_dir
    batch_size = FLAGS.batch_size
    bitdepth = FLAGS.bitdepth

    n_tokens = tokens.token_count(bitdepth=bitdepth)
    max_length = FLAGS.image_size ** 2
    model = trax.models.TransformerLM(n_tokens, max_len=max_length, mode='predict')

    example = np.zeros([batch_size, max_length]).astype(np.int32)
    signature = trax.shapes.signature(example)
    model.init_from_file(f"{output_dir}/model.pkl.gz", weights_only=True, input_signature=signature)
    #model.init_from_file(f"{output_dir}/model-88000.pkl.gz", weights_only=True, input_signature=signature)

    imgdb = GlobDatabase(FLAGS.images, "*.jpg")
    work_list = imgdb.select("train")
    train_itr = iter_dataset(work_list, batch_size=FLAGS.batch_size, group="train")
    work_list = imgdb.select("val")
    eval_itr = iter_dataset(work_list, batch_size=FLAGS.batch_size, group="val")

    batch = next(eval_itr)
    inp = batch[0]
    orig = np.copy(inp)
    fill = [[0]] * batch_size
    orig = np.append(orig, fill, axis=-1)
    orig = [tokens.tokens_to_image(ary, bitdepth=bitdepth) for ary in orig]

    seed_len = max_length // 2
    fill_len = max_length - seed_len
    top = inp[:, :max_length // 2]
    bottom = np.zeros((batch_size, fill_len), dtype=np.int32)
    inp = np.append(top, bottom, axis=-1)
    inp = jnp.array(inp)

    #tok_images = greedy_search(inp, model, seed_len, max_length)

    """
    tok_images = trax.supervised.decoding.autoregressive_sample(
            model, 
            inp, 
            start_id=-1,
            eos_id=-1,
            batch_size=batch_size,
            temperature=0.1,
            max_length=max_length)
    """

    tok_images = inp
    for (idx, gen) in enumerate(tok_images):
        gen = tokens.tokens_to_image(gen, bitdepth=bitdepth)
        gen = gen.resize((512, 512))
        gen.save(f"example-{1 + idx:03d}.png")
        seed = orig[idx]
        seed = seed.resize((512, 512))
        seed.save(f"seed-{1 + idx:03d}.png")

