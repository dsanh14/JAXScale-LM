"""Train the BPE tokenizer declared by a config (no-op for byte tokenizers).

Example:
    uv run python scripts/train_tokenizer.py --config configs/train/single_device.yaml
"""

from __future__ import annotations

import argparse

from jaxscale_lm.config import load_config
from jaxscale_lm.data.dataset import load_documents
from jaxscale_lm.data.tokenizer import train_bpe_tokenizer
from jaxscale_lm.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.logging.level, config.logging.json_format)

    if config.tokenizer.kind == "byte":
        print("tokenizer.kind is 'byte': nothing to train (fixed 259-entry vocab).")
        return
    if config.tokenizer.path is None:
        raise SystemExit(
            "tokenizer.kind is 'bpe' but tokenizer.path is unset; add it to the config."
        )

    docs = load_documents(config.data, config.project.seed)
    tokenizer = train_bpe_tokenizer(docs, config.tokenizer.vocab_size, config.tokenizer.path)
    print(f"trained BPE tokenizer: vocab_size={tokenizer.vocab_size}")
    print(f"saved to:              {config.tokenizer.path}")


if __name__ == "__main__":
    main()
