"""Download (or generate) the dataset referenced by a config, then report stats.

Examples:
    uv run python scripts/download_data.py --config configs/train/cpu_smoke.yaml
    uv run python scripts/download_data.py --config configs/train/single_device.yaml
"""

from __future__ import annotations

import argparse

from jaxscale_lm.config import load_config
from jaxscale_lm.data.dataset import load_documents, split_documents
from jaxscale_lm.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.logging.level, config.logging.json_format)

    docs = load_documents(config.data, config.project.seed)
    splits = split_documents(docs, config.data.validation_fraction, config.project.seed)
    total_chars = sum(len(d) for d in docs)
    print(f"source:               {config.data.source}")
    print(f"documents:            {len(docs)}")
    print(f"total characters:     {total_chars:,}")
    print(f"train documents:      {len(splits.train)}")
    print(f"validation documents: {len(splits.validation)}")
    if config.data.source == "tinystories":
        print(f"cache directory:      {config.data.cache_dir}")


if __name__ == "__main__":
    main()
