"""Explicit device placement of state and batches onto the mesh."""

from __future__ import annotations

import jax
from jax.sharding import NamedSharding

from jaxscale_lm.types import Batch, PyTree


def place_tree(tree: PyTree, sharding: NamedSharding) -> PyTree:
    """Place every array leaf of a pytree with the given sharding."""
    return jax.device_put(tree, sharding)


def place_batch(batch: Batch, sharding: NamedSharding) -> Batch:
    """Place a batch's arrays with the given sharding."""
    return Batch(*(jax.device_put(x, sharding) for x in batch))
