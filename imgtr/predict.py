import trax
import jax

import numpy as np
from jax import numpy as jnp
from trax import layers as tl
from absl import logging

from . import tokens
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

    bitdepth = [int(x) for x in FLAGS.bitdepth.split(',')]
    assert len(bitdepth) == 3

    vocab_size = tokens.token_count(bitdepth=bitdepth)
    max_length = FLAGS.image_size ** 2
    model = trax.models.TransformerLM(vocab_size, max_len=max_length, mode='predict')

    imgdb = ImageDatabase(FLAGS.images)
    train_itr = batch_dataset(imgdb, batch_size=batch_size, group="train")
    eval_itr = batch_dataset(imgdb, batch_size=batch_size, group="val")

    example = np.zeros([batch_size, max_length]).astype(np.int32)
    signature = trax.shapes.signature(example)
    #model.init_from_file(f"{output_dir}/model.pkl.gz", weights_only=True, input_signature=signature)
    model.init_from_file(f"{output_dir}/model-88000.pkl.gz", weights_only=True, input_signature=signature)

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

    tok_images = greedy_search(inp, model, seed_len, max_length)

    """
    tok_images = trax.supervised.decoding.autoregressive_sample(
            model, 
            batch, 
            start_id=batch[0][-1],
            eos_id=-1,
            temperature=0,
            max_length=max_length)
    """

    #images = [tokens.tokens_to_image(ary) for ary in tok_images]
    for (idx, gen) in enumerate(tok_images):
        gen = tokens.tokens_to_image(gen, bitdepth=bitdepth)
        gen = gen.resize((512, 512))
        gen.save(f"example-{1 + idx:03d}.png")
        seed = orig[idx]
        seed = seed.resize((512, 512))
        seed.save(f"seed-{1 + idx:03d}.png")
