import jax.numpy as jnp
from jax import lax

import trax
from trax import layers as tl

# ported over from tip of trax repo
#
def one_hot(x, n_categories, dtype=jnp.float32):
  indices_less_than_n = jnp.arange(n_categories)
  return jnp.array(x[..., jnp.newaxis] == indices_less_than_n, dtype)

def log_softmax(x, axis=-1):
  return x - trax.fastmath.logsumexp(x, axis=axis, keepdims=True)

def _category_cross_entropy(model_output, targets):
  target_distributions = one_hot(targets, model_output.shape[-1])
  model_log_distributions = log_softmax(model_output)
  return - jnp.sum(target_distributions * model_log_distributions, axis=-1)

def WeightedCategoryAccuracy():
  def f(model_output, targets, weights):
    predictions = jnp.argmax(model_output, axis=-1)
    trax.shapes.assert_same_shape(predictions, targets)
    ones_and_zeros = jnp.equal(predictions, targets)
    return jnp.sum(ones_and_zeros * weights) / jnp.sum(weights)
  return tl.Fn('WeightedCategoryAccuracy', f)

def WeightedCategoryCrossEntropy():
  def f(model_output, targets, weights):
    cross_entropies = _category_cross_entropy(model_output, targets)
    return jnp.sum(cross_entropies * weights) / jnp.sum(weights)
  return tl.Fn('WeightedCategoryCrossEntropy', f)


