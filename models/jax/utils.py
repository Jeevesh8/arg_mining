from typing import Tuple

import jax
import jax.numpy as jnp


def add_garbage_dims(array):
    """Adds extra slice at last of every dimension, filled with zeros."""
    """FOR-LOOP equivalent
    for i, _ in enumerate(array.shape):
        array = jnp.concatenate(
            [
                array,
                jnp.expand_dims(
                    jnp.zeros(array.shape[:i] + array.shape[i + 1:],
                              dtype=array.dtype),
                    axis=i,
                ),
            ],
            axis=i,
        )
    return array
    """
    return jnp.pad(array, pad_width=tuple((0, 1) for _ in jnp.shape(array)))


def remove_garbage_dims(array):
    """Removes extra slice at last of every dimension, filled with zeros,
    which were added by add_garbage_dims()"""
    for i, _ in enumerate(array.shape):
        array = jnp.take(array, jnp.arange(array.shape[i] - 1), axis=i)
    return array
