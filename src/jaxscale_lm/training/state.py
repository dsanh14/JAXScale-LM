"""Training state: parameters, optimizer state, step counter, RNG key.

The model is handled functionally: ``nnx.split`` separates the static graph
definition from the parameter state, and jitted steps operate on the state
pytree only. ``TrainState`` is a NamedTuple, so it is itself a pytree that
moves through ``jax.jit`` and Orbax untouched.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import optax
from flax import nnx

from jaxscale_lm.types import PRNGKey


class TrainState(NamedTuple):
    """Everything that changes during training (the checkpointable pytree)."""

    params: nnx.State
    opt_state: optax.OptState
    step: jax.Array  # scalar int32
    rng_key: PRNGKey  # root key for dropout streams


def create_train_state(
    model: nnx.Module, tx: optax.GradientTransformation, rng_key: PRNGKey
) -> tuple[nnx.GraphDef, TrainState]:
    """Split the model and initialize optimizer state.

    Returns:
        ``(graphdef, state)`` — graphdef is static (hashable) and is closed
        over by the jitted step functions; state is the mutable pytree.
    """
    graphdef, params = nnx.split(model)
    opt_state = tx.init(params)
    return graphdef, TrainState(
        params=params,
        opt_state=opt_state,
        step=jax.numpy.zeros((), jax.numpy.int32),
        rng_key=rng_key,
    )
