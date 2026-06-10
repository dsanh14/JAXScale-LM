"""Shared type aliases and simple data containers."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import numpy as np

PyTree = Any
PRNGKey = jax.Array

# Batches are produced on the host as NumPy and placed on device by the
# trainer, so fields accept either array kind.
ArrayLike = jax.Array | np.ndarray


class Batch(NamedTuple):
    """A single next-token-prediction batch.

    Attributes:
        input_ids:  ``[batch, seq_len]`` int32 token ids fed to the model.
        target_ids: ``[batch, seq_len]`` int32 ids shifted one position left
            relative to ``input_ids`` (the next-token targets).
        loss_mask:  ``[batch, seq_len]`` float32; 1.0 where the target is a
            real token that should contribute to the loss, 0.0 otherwise.
    """

    input_ids: ArrayLike
    target_ids: ArrayLike
    loss_mask: ArrayLike
