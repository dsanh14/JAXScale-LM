"""Named shardings for the training and inference data structures.

Default partitioning strategy:

- **Parameters / optimizer state / RNG key**: replicated (every device holds
  a full copy). With pure data parallelism this is the correct placement;
  XLA inserts the gradient all-reduce automatically because replicated
  outputs must be consistent.
- **Batches**: sharded along the batch dimension over the ``data`` axis.
  Stacked train batches are ``[accum, micro, seq]`` and shard on axis 1.
"""

from __future__ import annotations

from jax.sharding import Mesh, NamedSharding, PartitionSpec

from jaxscale_lm.config import DistributedConfig


def replicated(mesh: Mesh) -> NamedSharding:
    """Full replication across the mesh."""
    return NamedSharding(mesh, PartitionSpec())


def train_batch_sharding(mesh: Mesh, config: DistributedConfig) -> NamedSharding:
    """Stacked train batches ``[accum, micro, seq]``: shard micro over data."""
    data_axis = config.axis_names[0]
    return NamedSharding(mesh, PartitionSpec(None, data_axis, None))


def eval_batch_sharding(mesh: Mesh) -> NamedSharding:
    """Eval batches are replicated: the final batch of a split can be ragged
    (not divisible by the data-axis size), and eval is a tiny fraction of
    runtime, so correctness wins over parallel speedup here."""
    return NamedSharding(mesh, PartitionSpec())


def validate_batch_divisibility(microbatch_size: int, mesh: Mesh, config: DistributedConfig) -> None:
    """Fail fast if the microbatch can't be evenly sharded over the data axis."""
    data_axis = config.axis_names[0]
    data_size = mesh.shape[data_axis]
    if microbatch_size % data_size != 0:
        raise ValueError(
            f"data.batch_size ({microbatch_size}) must be divisible by the data-"
            f"parallel axis size ({data_size}) so every device gets equal rows. "
            f"Use a batch size that is a multiple of {data_size}."
        )
