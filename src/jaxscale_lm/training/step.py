"""Jitted train and eval steps.

Purity contract: the functions built here are closed over *static* values
only (graphdef, optimizer, accumulation count). Everything dynamic — params,
optimizer state, batch, RNG key — flows through arguments, so the jitted
functions are pure and compile exactly once per input shape.

Gradient accumulation semantics: a step receives ``accum_steps``
microbatches stacked on a leading axis and scans over them, summing raw
(unnormalized) gradients and valid-token counts. The final update divides
the gradient sum by the *total* valid tokens, which makes the accumulated
update mathematically identical to one large-batch update (verified by a
regression test).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from jaxscale_lm.training.loss import LossStats, loss_stats
from jaxscale_lm.training.state import TrainState
from jaxscale_lm.types import Batch

TrainStepFn = Callable[[TrainState, Batch], tuple[TrainState, dict[str, jax.Array]]]
EvalStepFn = Callable[[nnx.State, Batch], LossStats]


def make_train_step(
    graphdef: nnx.GraphDef,
    tx: optax.GradientTransformation,
    schedule: optax.Schedule,
    accum_steps: int,
) -> TrainStepFn:
    """Build the jitted training step.

    The returned function expects batch arrays shaped
    ``[accum_steps, microbatch, seq]`` (stacked microbatches).
    """
    if accum_steps <= 0:
        raise ValueError(f"accum_steps must be positive, got {accum_steps}")

    def grads_for_microbatch(
        params: nnx.State, batch: Batch, key: jax.Array
    ) -> tuple[Any, LossStats]:
        def loss_fn(p: nnx.State) -> tuple[jax.Array, LossStats]:
            model = nnx.merge(graphdef, p)
            logits, _ = model(
                batch.input_ids, deterministic=False, rngs=nnx.Rngs(dropout=key)
            )
            stats = loss_stats(logits, batch.target_ids, batch.loss_mask)
            # Differentiate the *sum* so accumulated grads add linearly.
            return stats.nll_sum, stats

        grads, stats = jax.grad(loss_fn, has_aux=True)(params)
        return grads, stats

    def step(state: TrainState, batch: Batch) -> tuple[TrainState, dict[str, jax.Array]]:
        # Per-step dropout key derived from the root key; the root key itself
        # is never consumed, so restoring (rng_key, step) resumes the exact
        # stream an uninterrupted run would produce.
        step_key = jax.random.fold_in(state.rng_key, state.step)
        micro_keys = jax.random.split(step_key, accum_steps)

        if accum_steps == 1:
            squeezed = Batch(*(x[0] for x in batch))
            grads, stats = grads_for_microbatch(state.params, squeezed, micro_keys[0])
        else:

            def body(carry: tuple[Any, LossStats], xs: tuple[Batch, jax.Array]):
                grads_acc, stats_acc = carry
                mb, key = xs
                grads, stats = grads_for_microbatch(state.params, mb, key)
                grads_acc = jax.tree.map(jnp.add, grads_acc, grads)
                stats_acc = LossStats(*(a + b for a, b in zip(stats_acc, stats, strict=True)))
                return (grads_acc, stats_acc), None

            zero_grads = jax.tree.map(jnp.zeros_like, state.params)
            zero_stats = LossStats(
                nll_sum=jnp.zeros((), jnp.float32),
                correct_sum=jnp.zeros((), jnp.float32),
                valid_tokens=jnp.zeros((), jnp.float32),
            )
            (grads, stats), _ = jax.lax.scan(
                body, (zero_grads, zero_stats), (batch, micro_keys)
            )

        # Normalize by total valid tokens across all microbatches.
        denom = jnp.maximum(stats.valid_tokens, 1.0)
        grads = jax.tree.map(lambda g: g / denom, grads)

        updates, opt_state = tx.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        new_state = TrainState(
            params=params,
            opt_state=opt_state,
            step=state.step + 1,
            rng_key=state.rng_key,
        )
        metrics = {
            "loss": stats.nll_sum / denom,
            "accuracy": stats.correct_sum / denom,
            "valid_tokens": stats.valid_tokens,
            "grad_norm": optax.global_norm(grads),
            "learning_rate": schedule(state.step),
        }
        return new_state, metrics

    return step


def make_eval_step(graphdef: nnx.GraphDef) -> EvalStepFn:
    """Build the jitted evaluation step (no dropout, no state mutation)."""

    def step(params: nnx.State, batch: Batch) -> LossStats:
        model = nnx.merge(graphdef, params)
        logits, _ = model(batch.input_ids, deterministic=True)
        return loss_stats(logits, batch.target_ids, batch.loss_mask)

    return step
