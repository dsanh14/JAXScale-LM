"""Train (or resume) a JAXScale-LM model.

Examples:
    uv run python scripts/train.py --config configs/train/cpu_smoke.yaml
    uv run python scripts/train.py --config configs/train/single_device.yaml --resume latest
    uv run python scripts/train.py --config configs/train/single_device.yaml --resume 200
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument(
        "--resume",
        default=None,
        metavar="latest|STEP",
        help="Resume from the latest checkpoint or a specific step number",
    )
    args = parser.parse_args()

    from jaxscale_lm.config import load_config
    from jaxscale_lm.training.trainer import Trainer
    from jaxscale_lm.utils.logging import setup_logging

    config = load_config(args.config)
    setup_logging(config.logging.level, config.logging.json_format)

    trainer = Trainer(config)
    if args.resume is not None:
        step = None if args.resume == "latest" else int(args.resume)
        trainer.resume(step)

    summary = trainer.train()
    print("final evaluation:")
    for key, value in summary.items():
        print(f"  {key}: {value:.6g}")
    print(f"checkpoints: {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
