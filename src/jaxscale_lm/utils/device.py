"""Device inspection utilities.

Everything here is read-only introspection of the JAX runtime; nothing
allocates device memory or triggers compilation.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax

from jaxscale_lm.config import DTypeName


@dataclass(frozen=True)
class DeviceReport:
    """Snapshot of the visible JAX device topology."""

    platform: str
    device_kinds: tuple[str, ...]
    device_ids: tuple[int, ...]
    process_index: int
    process_count: int
    local_device_count: int
    global_device_count: int

    def lines(self) -> list[str]:
        return [
            f"platform:            {self.platform}",
            f"process index/count: {self.process_index}/{self.process_count}",
            f"local devices:       {self.local_device_count}",
            f"global devices:      {self.global_device_count}",
            f"device ids:          {list(self.device_ids)}",
            f"device kinds:        {list(dict.fromkeys(self.device_kinds))}",
        ]


def device_report() -> DeviceReport:
    """Collect the current device topology."""
    devices = jax.devices()
    return DeviceReport(
        platform=jax.default_backend(),
        device_kinds=tuple(d.device_kind for d in devices),
        device_ids=tuple(d.id for d in devices),
        process_index=jax.process_index(),
        process_count=jax.process_count(),
        local_device_count=jax.local_device_count(),
        global_device_count=jax.device_count(),
    )


def supports_dtype(dtype: DTypeName) -> bool:
    """Whether the current backend can run the given compute dtype safely.

    float16 compute is rejected on CPU/TPU: CPU emulation is slow and not
    representative, and TPUs prefer bfloat16. bfloat16 and float32 work on
    every backend JAX supports.
    """
    if dtype == "float16":
        return jax.default_backend() == "gpu"
    return True
