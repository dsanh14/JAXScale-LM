"""Evaluate a checkpoint on its validation split.

Example:
    uv run python scripts/evaluate.py --checkpoint artifacts/checkpoints/cpu_smoke/latest
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint run directory, optionally suffixed with /latest or /<step>",
    )
    parser.add_argument(
        "--num-batches", type=int, default=None, help="Override evaluation.num_batches"
    )
    args = parser.parse_args()

    from jaxscale_lm.config import Config
    from jaxscale_lm.training.checkpoint import read_metadata, resolve_checkpoint
    from jaxscale_lm.training.trainer import Trainer
    from jaxscale_lm.utils.logging import setup_logging

    ref = resolve_checkpoint(args.checkpoint)
    step, metadata = read_metadata(ref.root, ref.step)
    config = Config.model_validate(metadata["config"])
    # Restore from the directory the user pointed at, not whatever path the
    # saved config recorded (the artifacts tree may have moved).
    config = config.model_copy(
        update={"checkpoint": config.checkpoint.model_copy(update={"directory": ref.root})}
    )
    if args.num_batches is not None:
        config = config.model_copy(
            update={
                "evaluation": config.evaluation.model_copy(update={"num_batches": args.num_batches})
            }
        )
    setup_logging(config.logging.level, config.logging.json_format)

    trainer = Trainer(config)
    try:
        trainer.resume(step)
        summary = trainer.evaluate()
    finally:
        trainer.checkpointer.close()

    print(f"checkpoint step {step}:")
    for key, value in summary.items():
        print(f"  {key}: {value:.6g}")


if __name__ == "__main__":
    main()
