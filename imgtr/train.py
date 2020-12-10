# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sequence Tagging example.

This script trains a Transformer on the Universal dependency dataset.
"""

import tensorflow as tf

import functools
import os
import time
from absl import app
from absl import flags
from absl import logging
from jax import random
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import multiprocessing as mp

from flax import jax_utils
from flax import linen as nn
from flax import optim
from flax.metrics import tensorboard
from flax.training import common_utils

from . import transformer
from . import tokens
from . imgdb import ImageDatabase

FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', default='', help=('Directory for model data.'))

flags.DEFINE_string('experiment', default='imgtr', help=('Experiment name.'))


flags.DEFINE_integer( 'batch_size', default=16, help=('Batch size for training.'))

flags.DEFINE_integer( 'image_size', default=22, help=('Edge size for square image.'))

flags.DEFINE_string('bitdepth', default='5,4,4', help=('HSV bitdepths'))

flags.DEFINE_string('images', default="images", help=('Path to images used for training and validation'))

flags.DEFINE_integer(
    'eval_frequency',
    default=100,
    help=('Frequency of eval during training, e.g. every 1000 steps.'))

flags.DEFINE_integer(
    'num_train_steps', default=75000, help=('Number of train steps.'))

flags.DEFINE_float('learning_rate', default=0.05, help=('Learning rate.'))

flags.DEFINE_float(
    'weight_decay',
    default=1e-1,
    help=('Decay factor for AdamW style weight decay.'))

flags.DEFINE_integer(
    'random_seed', default=0, help=('Integer for PRNG random seed.'))

def create_learning_rate_scheduler(
    factors='constant * linear_warmup * rsqrt_decay',
    base_learning_rate=0.5,
    warmup_steps=8000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000):
  """creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: a string with factors separated by '*' that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: how many steps to warm up for in the warmup schedule.
    decay_factor: The amount to decay the learning rate by.
    steps_per_decay: How often to decay the learning rate.
    steps_per_cycle: Steps per cycle when using cosine decay.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split('*')]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == 'constant':
        ret *= base_learning_rate
      elif name == 'linear_warmup':
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'rsqrt_normalized_decay':
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'decay_every':
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == 'cosine_decay':
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError('Unknown factor %s.' % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


def compute_weighted_cross_entropy(logits, targets, weights=None):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch x length]

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  onehot_targets = common_utils.onehot(targets, logits.shape[-1])
  loss = -jnp.sum(onehot_targets * nn.log_softmax(logits), axis=-1)
  normalizing_factor = onehot_targets.sum()
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_weighted_accuracy(logits, targets, weights=None):
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch x length]

  Returns:
    Tuple of scalar accuracy and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  normalizing_factor = np.prod(logits.shape[:-1])
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_metrics(logits, labels, weights):
  """Compute summary metrics."""
  loss, weight_sum = compute_weighted_cross_entropy(logits, labels, weights)
  acc, _ = compute_weighted_accuracy(logits, labels, weights)
  metrics = {
      'loss': loss,
      'accuracy': acc,
      'denominator': weight_sum,
  }
  metrics = np.sum(metrics, -1)
  return metrics

def loss_fn(params, model, batch, dropout_rng):
    """Loss function used for training."""
    (inputs, targets, weights) = batch
    logits = model.apply({'params': params}, inputs=inputs, train=True,
                         rngs={'dropout': dropout_rng})
    loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights)
    mean_loss = loss / weight_sum
    return mean_loss, logits

def train_step(optimizer, batch, learning_rate, model, dropout_rng=None):
  (inputs, targets, weights) = batch
  dropout_rng, new_dropout_rng = random.split(dropout_rng)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target, model, batch, dropout_rng)
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=learning_rate)
  metrics = compute_metrics(logits, targets, weights)
  metrics['learning_rate'] = learning_rate
  return new_optimizer, metrics, new_dropout_rng

def _iden(it):
    return it

def batch_dataset(imgdb, batch_size=1, group=None):
    def _inner(imgdb=imgdb, group=group):
        while True:
            cur = imgdb.get_cursor(group=group)
            for img in cur:
                img = img()
                toks = tokens.image_to_tokens(img, size=FLAGS.image_size)
                x = jnp.array(toks[:-1])
                y = jnp.array(toks[1:])
                w = jnp.ones_like(x).astype(np.float)
                yield (x, y, w)

    def _batch(itr, batch_size=batch_size):
        while True:
            batch = [next(itr) for i in range(batch_size)]
            cols = zip(*batch)
            batch = [jnp.vstack(it) for it in cols]
            yield batch

    def _outer(imgdb, batch_size=1, group=None):
        pool = mp.Pool()
        itr = _batch(_inner(imgdb, group), batch_size)
        func = lambda it: it
        batch_itr = pool.imap_unordered(_iden, itr)
        for batch in batch_itr:
            yield batch
    
    def _debug():
        max_length = FLAGS.image_size ** 2
        ary = jnp.ones((batch_size, max_length))
        while True:
            yield (ary, ary, ary)
    return _debug()
    #return _outer(imgdb, batch_size, group)

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Make sure tf does not allocate gpu memory.
  tf.config.experimental.set_visible_devices([], 'GPU')

  batch_size = FLAGS.batch_size
  learning_rate = FLAGS.learning_rate
  num_train_steps = FLAGS.num_train_steps
  eval_freq = FLAGS.eval_frequency
  random_seed = FLAGS.random_seed

  if batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  device_batch_size = batch_size // jax.device_count()
  bitdepth = [int(x) for x in FLAGS.bitdepth.split(',')]
  assert len(bitdepth) == 3

  train_summary_writer = tensorboard.SummaryWriter(
      os.path.join(FLAGS.model_dir, FLAGS.experiment + '_train'))
  eval_summary_writer = tensorboard.SummaryWriter(
      os.path.join(FLAGS.model_dir, FLAGS.experiment + '_eval'))

  # create the training and development dataset
  vocab_size = tokens.token_count(bitdepth=bitdepth)
  max_length = FLAGS.image_size ** 2
  config = transformer.TransformerConfig(
      vocab_size=vocab_size,
      output_vocab_size=vocab_size,
      max_len=max_length)
  print(config)
  model = transformer.Transformer(config)

  imgdb = ImageDatabase(FLAGS.images)
  train_iter = batch_dataset(imgdb, batch_size=FLAGS.batch_size, group="train")
  eval_iter = batch_dataset(imgdb, batch_size=FLAGS.batch_size, group="eval")

  rng = random.PRNGKey(random_seed)
  rng, init_rng = random.split(rng)

  init_batch = jnp.ones((1, config.max_len), jnp.float32)
  init_variables = model.init(init_rng, inputs=init_batch, train=False)

  optimizer_def = optim.Adam(learning_rate, beta1=0.9, beta2=0.98,
      eps=1e-9, weight_decay=1e-1)
  optimizer = optimizer_def.create(init_variables['params'])

  learning_rate_fn = create_learning_rate_scheduler(
      base_learning_rate=learning_rate)

  def eval_step(params, batch):
    """Calculate evaluation metrics on a batch."""
    (inputs, targets, weights) = batch
    logits = model.apply({'params': params}, inputs=inputs, train=False)
    return compute_metrics(logits, targets, weights)

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap'd training update for performance.
  (rng, dropout_rngs) = random.split(rng)
  metrics_all = []
  tick = time.time()
  best_dev_score = 0
  for (step, batch) in enumerate(train_iter):
    learning_rate = learning_rate_fn(step)
    (optimizer, metrics, dropout_rngs) = train_step(optimizer, batch, learning_rate, model, dropout_rng=dropout_rngs) 
    metrics_all.append(metrics)

    if (step + 1) % eval_freq == 0:
      metrics_all = pd.DataFrame(metrics_all)
      summary = metrics_all.mean().to_dict()
      logging.info('train in step: %d, loss: %.4f', step, summary['loss'])
      tock = time.time()
      steps_per_sec = eval_freq / (tock - tick)
      tick = tock
      train_summary_writer.scalar('steps per second', steps_per_sec, step)
      for key, val in summary.items():
        train_summary_writer.scalar(key, val, step)
      train_summary_writer.flush()
      metrics_all = [] 

      eval_metrics = []
      for eval_batch in eval_iter:
        metrics = eval_step(optimizer.target, eval_batch)
        eval_metrics.append(metrics)
      metrics_all = pd.DataFrame(metrics_all)
      eval_summary = metrics_all.mean().to_dict()

      logging.info('eval in step: %d, loss: %.4f, accuracy: %.4f', step, eval_summary['loss'], eval_summary['accuracy'])

      if best_dev_score < eval_summary['accuracy']:
        best_dev_score = eval_summary['accuracy']
        # TODO: save model.
      eval_summary['best_dev_score'] = best_dev_score
      logging.info('best development model score %.4f', best_dev_score)
      for key, val in eval_summary.items():
        eval_summary_writer.scalar(key, val, step)
      eval_summary_writer.flush()

def run():
  app.run(main)
