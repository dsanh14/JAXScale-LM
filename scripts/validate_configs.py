"""Validate every shipped YAML config (defaults composition + cross-checks).

Examples:
    uv run python scripts/validate_configs.py
    uv run python scripts/validate_configs.py --configs-dir configs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("configs"),
        help="Directory scanned recursively for *.yaml configs",
    )
    args = parser.parse_args()

    from jaxscale_lm.config import load_config

    paths = sorted(args.configs_dir.rglob("*.yaml"))
    if not paths:
        print(f"error: no *.yaml configs found under {args.configs_dir}/", file=sys.stderr)
        raise SystemExit(1)

    for path in paths:
        config = load_config(path)
        print(
            f"config OK: {path} "
            f"(run_name={config.project.run_name}, "
            f"{config.model.num_layers}L x {config.model.hidden_size}h)"
        )
    print(f"validated {len(paths)} configs")


if __name__ == "__main__":
    main()
