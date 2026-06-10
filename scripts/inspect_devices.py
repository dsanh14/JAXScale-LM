"""Print the JAX device topology visible to this process.

Examples:
    uv run python scripts/inspect_devices.py
    uv run python scripts/inspect_devices.py --simulate-devices 8
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulate-devices",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Simulate N CPU devices via "
            "XLA_FLAGS=--xla_force_host_platform_device_count=N. "
            "These are host CPU devices for development only, not accelerators."
        ),
    )
    args = parser.parse_args()

    if args.simulate_devices is not None:
        if args.simulate_devices <= 0:
            parser.error(f"--simulate-devices must be positive, got {args.simulate_devices}")
        # Must be set before JAX initializes; that's why jax is imported below.
        flags = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = (
            f"{flags} --xla_force_host_platform_device_count={args.simulate_devices}".strip()
        )

    from jaxscale_lm.utils.device import device_report

    report = device_report()
    for line in report.lines():
        print(line)
    if args.simulate_devices is not None:
        print(
            f"note: {args.simulate_devices} simulated CPU devices "
            "(--xla_force_host_platform_device_count); not real accelerators."
        )


if __name__ == "__main__":
    main()
