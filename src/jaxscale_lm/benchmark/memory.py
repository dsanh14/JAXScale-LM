"""Best-effort memory measurement.

Host and device memory are reported separately and every number carries its
source. When a backend exposes no statistics (CPU device memory, notably)
the value is reported as ``"unsupported"`` — never invented or substituted
with a host-side proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax


@dataclass(frozen=True)
class MemorySnapshot:
    """Memory readings with provenance."""

    host_rss_bytes: int | None
    host_source: str
    device_stats: dict[str, Any] | str  # per-device stats or "unsupported"
    device_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "host_rss_bytes": self.host_rss_bytes,
            "host_source": self.host_source,
            "device_stats": self.device_stats,
            "device_source": self.device_source,
        }


@dataclass
class _DeviceStatsResult:
    stats: dict[str, Any] = field(default_factory=dict)
    supported: bool = False


def _collect_device_stats() -> _DeviceStatsResult:
    result = _DeviceStatsResult()
    for device in jax.local_devices():
        getter = getattr(device, "memory_stats", None)
        if getter is None:
            continue
        stats = getter()
        if stats:  # CPU devices return None/empty
            result.stats[f"{device.platform}:{device.id}"] = {
                key: stats[key]
                for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
                if key in stats
            }
            result.supported = True
    return result


def snapshot() -> MemorySnapshot:
    """Take a memory snapshot of this process and its local devices."""
    try:
        import psutil

        rss: int | None = psutil.Process().memory_info().rss
        host_source = "psutil.Process().memory_info().rss"
    except ImportError:
        rss = None
        host_source = "unavailable (psutil not installed)"

    device = _collect_device_stats()
    if device.supported:
        return MemorySnapshot(
            host_rss_bytes=rss,
            host_source=host_source,
            device_stats=device.stats,
            device_source="jax Device.memory_stats()",
        )
    return MemorySnapshot(
        host_rss_bytes=rss,
        host_source=host_source,
        device_stats="unsupported",
        device_source=f"backend '{jax.default_backend()}' exposes no device memory stats",
    )
