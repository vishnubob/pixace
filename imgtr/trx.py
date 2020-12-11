import os
import sys
import operator
import functools
import shutil
import multiprocessing as mp

from PIL import Image
from skimage import color
import numpy as np
import trax
from trax.supervised import training
from trax.models import transformer
from jax import lax
from jax import numpy as jnp
from trax import layers as tl

from absl import app
from absl import flags
from absl import logging
from . import tokens
from . imgdb import ImageDatabase

FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', default='model', help=('Directory for model data.'))
flags.DEFINE_string('experiment', default='imgtr', help=('Experiment name.'))
flags.DEFINE_integer('batch_size', default=16, help=('Batch size for training.'))
flags.DEFINE_integer('image_size', default=22, help=('Edge size for square image.'))
flags.DEFINE_string('bitdepth', default='5,4,4', help=('HSV bitdepths'))
flags.DEFINE_string('images', default="images", help=('Path to images used for training and validation'))
flags.DEFINE_integer('epochs', default=100, help=('number of epochs'))
flags.DEFINE_integer('steps_per_epoch', default=1000, help=('number of epochs'))

def backup_checkpoint(output_dir, training_loop):
    old_path = os.path.join(output_dir, f"model.pkl.gz")
    if not os.path.exists(old_path):
        return
    new_path = os.path.join(output_dir, f"model-{training_loop.step:05d}.pkl.gz")
    shutil.copyfile(old_path, new_path)

def _iden(it):
    return it

def batch_dataset(imgdb, batch_size=1, group=None):
    def _inner(imgdb=imgdb, group=group):
        while True:
            cur = imgdb.get_cursor(group=group)
            for img in cur:
                img = img()
                toks = tokens.image_to_tokens(img, size=FLAGS.image_size)
                x = jnp.array(toks[:-1]).astype(jnp.int32)
                y = jnp.array(toks[1:]).astype(jnp.int32)
                w = jnp.ones_like(x).astype(np.float)
                yield (x, y, w)

    def _batch(itr, batch_size=batch_size):
        while True:
            batch = [next(itr) for i in range(batch_size)]
            cols = zip(*batch)
            batch = [jnp.vstack(it) for it in cols]
            yield batch

    def _outer(imgdb, batch_size=1, group=None):
        pool = mp.Pool(4)
        batch_itr = _batch(_inner(imgdb, group), batch_size)
        batch_itr = pool.imap_unordered(_iden, batch_itr)
        for batch in batch_itr:
            yield batch
    
    def _debug():
        max_length = FLAGS.image_size ** 2
        ary = jnp.ones((batch_size, max_length))
        while True:
            yield (ary.astype(jnp.int32), ary.astype(jnp.int32), ary)
    return _outer(imgdb, batch_size, group)
    #return _batch(_inner(imgdb, group), batch_size)
    #return _debug()

def train_model(argv):
    output_dir = FLAGS.model_dir
    batch_size = FLAGS.batch_size
    steps_per_epoch = FLAGS.steps_per_epoch

    bitdepth = [int(x) for x in FLAGS.bitdepth.split(',')]
    assert len(bitdepth) == 3

    # create the training and development dataset
    vocab_size = tokens.token_count(bitdepth=bitdepth)
    max_length = FLAGS.image_size ** 2
    model = trax.models.TransformerLM(vocab_size, max_len=max_length)

    imgdb = ImageDatabase(FLAGS.images)
    train_itr = batch_dataset(imgdb, batch_size=FLAGS.batch_size, group="train")
    eval_itr = batch_dataset(imgdb, batch_size=FLAGS.batch_size, group="val")

    train_task = training.TrainTask(
        labeled_data=train_itr,
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=trax.optimizers.Adam(0.01),
        n_steps_per_checkpoint=steps_per_epoch,
    )

    eval_task = training.EvalTask(
        labeled_data=eval_itr,
        metrics=[tl.CrossEntropyLoss(), tl.CrossEntropySum(), tl.Accuracy()],
        n_eval_batches=steps_per_epoch // 10
    )

    training_loop = training.Loop(
        model,
        train_task,
        eval_tasks=[eval_task],
        output_dir=output_dir
    )

    #lm_generate_example(training_loop, eval_batch, train_model, dc_train.captions)
    while True:
        training_loop.run(steps_per_epoch)
        backup_checkpoint(output_dir, training_loop)
        #lm_generate_example(training_loop, eval_batch, train_model, dc_train.captions)

def predict_model(argv):
    output_dir = FLAGS.model_dir
    #batch_size = FLAGS.batch_size
    batch_size = 1

    bitdepth = [int(x) for x in FLAGS.bitdepth.split(',')]
    assert len(bitdepth) == 3

    vocab_size = tokens.token_count(bitdepth=bitdepth)
    max_length = FLAGS.image_size ** 2
    model = trax.models.TransformerLM(vocab_size, max_len=max_length)

    imgdb = ImageDatabase(FLAGS.images)
    train_itr = batch_dataset(imgdb, batch_size=batch_size, group="train")
    eval_itr = batch_dataset(imgdb, batch_size=batch_size, group="val")

    example = np.zeros([batch_size, max_length]).astype(np.int32)
    signature = trax.shapes.signature(example)
    model.init_from_file(f"{output_dir}/model.pkl.gz", weights_only=True, input_signature=signature)
    #model.init_from_file(f"{output_dir}/model-88000.pkl.gz", weights_only=True, input_signature=signature)

    batch = next(eval_itr)
    orig = batch[0][:] 
    orig = [np.append(orig, [0])]
    orig = [tokens.tokens_to_image(ary) for ary in orig]

    batch = batch[0].T[:max_length // 2].T

    tok_images = trax.supervised.decoding.autoregressive_sample(
            model, 
            batch, 
            start_id=batch[0][-1],
            eos_id=-1,
            temperature=0,
            max_length=max_length)

    #images = [tokens.tokens_to_image(ary) for ary in tok_images]
    for (idx, gen) in enumerate(tok_images):
        print(gen)
        gen = tokens.tokens_to_image(gen)
        gen = gen.resize((512, 512))
        gen.save(f"example-{1 + idx:03d}.png")
        seed = orig[idx]
        seed = seed.resize((512, 512))
        seed.save(f"seed-{1 + idx:03d}.png")
        
    print(tok_images.shape)

def run():
    app.run(predict_model)
