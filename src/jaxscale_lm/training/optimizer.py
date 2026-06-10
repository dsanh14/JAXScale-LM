"""AdamW with warmup schedule, weight-decay masking, and gradient clipping."""

from __future__ import annotations

import jax
import optax

from jaxscale_lm.config import OptimizerConfig
from jaxscale_lm.types import PyTree


def build_schedule(config: OptimizerConfig, max_steps: int) -> optax.Schedule:
    """Linear warmup followed by cosine/linear decay (or constant)."""
    if config.warmup_steps >= max_steps:
        raise ValueError(
            f"optimizer.warmup_steps ({config.warmup_steps}) must be smaller than "
            f"training.max_steps ({max_steps})."
        )
    end = config.learning_rate * config.min_learning_rate_ratio
    decay_steps = max_steps - config.warmup_steps
    warmup = optax.linear_schedule(0.0, config.learning_rate, max(config.warmup_steps, 1))
    if config.schedule == "cosine":
        decay = optax.cosine_decay_schedule(
            config.learning_rate, decay_steps, alpha=config.min_learning_rate_ratio
        )
    elif config.schedule == "linear":
        decay = optax.linear_schedule(config.learning_rate, end, decay_steps)
    else:  # constant
        decay = optax.constant_schedule(config.learning_rate)
    if config.warmup_steps == 0:
        return decay
    return optax.join_schedules([warmup, decay], [config.warmup_steps])


def weight_decay_mask(params: PyTree) -> PyTree:
    """Apply weight decay only to leaves with rank >= 2.

    Matrices (attention/MLP projections, embeddings) are decayed; vectors and
    scalars (RMSNorm scales, biases) are not — decaying normalization scales
    pulls activations toward zero and hurts optimization.
    """
    return jax.tree.map(lambda leaf: getattr(leaf, "ndim", 0) >= 2, params)


def build_optimizer(
    config: OptimizerConfig, max_steps: int
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """Build the optax chain: global-norm clip -> AdamW(schedule, masked decay)."""
    schedule = build_schedule(config, max_steps)
    adamw = optax.adamw(
        learning_rate=schedule,
        b1=config.beta1,
        b2=config.beta2,
        eps=config.eps,
        weight_decay=config.weight_decay,
        mask=weight_decay_mask,
    )
    if config.grad_clip_norm is not None:
        tx = optax.chain(optax.clip_by_global_norm(config.grad_clip_norm), adamw)
    else:
        tx = adamw
    return tx, schedule
