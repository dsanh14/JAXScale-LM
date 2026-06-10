"""Human-readable diagnostics for devices, meshes, and array shardings."""

from __future__ import annotations

import jax
from jax.sharding import Mesh

from jaxscale_lm.utils.device import device_report


def describe_mesh(mesh: Mesh) -> list[str]:
    """Mesh topology lines for logging/CLI output."""
    return [
        f"mesh shape:          {dict(mesh.shape)}",
        f"mesh devices:        {[d.id for d in mesh.devices.flatten()]}",
        f"mesh axis names:     {list(mesh.axis_names)}",
    ]


def describe_array(name: str, array: jax.Array) -> list[str]:
    """Shape/sharding/addressable-shard lines for one array."""
    lines = [
        f"{name}.shape:        {tuple(array.shape)}",
        f"{name}.dtype:        {array.dtype}",
        f"{name}.sharding:     {array.sharding}",
    ]
    shards = getattr(array, "addressable_shards", None)
    if shards is not None:
        per_device = [(s.device.id, tuple(s.data.shape)) for s in shards]
        lines.append(f"{name}.shards:       {per_device}")
    return lines


def full_report(mesh: Mesh | None = None, sample: jax.Array | None = None) -> str:
    """Combined device/mesh/sample-array report."""
    lines = device_report().lines()
    if mesh is not None:
        lines += describe_mesh(mesh)
    if sample is not None:
        lines += describe_array("sample", sample)
    return "\n".join(lines)
