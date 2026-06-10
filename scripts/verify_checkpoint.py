"""Inspect and verify a checkpoint: metadata, steps, restorability.

Example:
    uv run python scripts/verify_checkpoint.py \
        --checkpoint artifacts/checkpoints/cpu_smoke/latest
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir, /latest, or /<step>")
    parser.add_argument(
        "--restore", action="store_true", help="Also fully restore the state (slower)"
    )
    args = parser.parse_args()

    from jaxscale_lm.config import Config
    from jaxscale_lm.training.checkpoint import read_metadata, resolve_checkpoint
    from jaxscale_lm.utils.logging import setup_logging

    setup_logging()
    ref = resolve_checkpoint(args.checkpoint)
    step, metadata = read_metadata(ref.root, ref.step)

    print(f"checkpoint root:   {ref.root}")
    print(f"step:              {step}")
    print(f"created at:        {metadata.get('created_at')}")
    print(f"jaxscale version:  {metadata.get('jaxscale_lm_version')}")
    print(f"jax version:       {metadata.get('jax_version')}")
    print(f"parameters:        {metadata.get('parameter_count'):,}")
    print(f"best metric:       {metadata.get('best_metric_name')} = {metadata.get('best_metric_value')}")
    model = metadata.get("model_config", {})
    print(
        f"model:             {model.get('num_layers')}L x {model.get('hidden_size')}h, "
        f"vocab {model.get('vocab_size')}, context {model.get('max_sequence_length')}"
    )

    if args.restore:
        from jaxscale_lm.inference.engine import InferenceEngine

        engine = InferenceEngine.from_checkpoint(args.checkpoint)
        config = Config.model_validate(metadata["config"])
        print(f"restore:           OK (step {engine.checkpoint_step}, "
              f"compute_dtype {config.model.compute_dtype})")


if __name__ == "__main__":
    main()
