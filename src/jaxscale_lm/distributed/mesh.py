"""Logical device mesh construction.

The mesh has two named axes:

- ``data``: batches are sharded along it (data parallelism).
- ``model``: reserved for tensor/model parallelism; size 1 by default. The
  axis exists so partition specs can reference it without special-casing,
  but no parameter sharding is enabled in the default configuration (see
  docs/sharding.md for the discussion and limitations).

A 1x1 mesh on a single device runs the exact same code path as a multi-
device mesh — sharding becomes replication and collectives become no-ops.
"""

from __future__ import annotations

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from jaxscale_lm.config import DistributedConfig


def build_mesh(config: DistributedConfig) -> Mesh:
    """Build the device mesh described by the config.

    ``data_axis_size == -1`` means "all devices not used by the model axis".

    Raises:
        ValueError: when the requested axis sizes don't match the available
            device count, with the actual topology in the message.
    """
    n_devices = jax.device_count()
    model_size = config.model_axis_size
    data_size = config.data_axis_size

    if model_size > n_devices:
        raise ValueError(
            f"distributed.model_axis_size={model_size} exceeds the {n_devices} "
            f"available device(s) on platform '{jax.default_backend()}'."
        )
    if data_size == -1:
        if n_devices % model_size != 0:
            raise ValueError(
                f"Cannot infer data axis: {n_devices} device(s) is not divisible by "
                f"distributed.model_axis_size={model_size}."
            )
        data_size = n_devices // model_size
    if data_size * model_size != n_devices:
        raise ValueError(
            f"Mesh of {data_size} (data) x {model_size} (model) = "
            f"{data_size * model_size} devices does not match the {n_devices} "
            f"available device(s). Adjust distributed.data_axis_size / "
            f"model_axis_size, or simulate devices with "
            f"XLA_FLAGS=--xla_force_host_platform_device_count=N (CPU dev only)."
        )

    devices = mesh_utils.create_device_mesh((data_size, model_size))
    return Mesh(devices, axis_names=config.axis_names)
