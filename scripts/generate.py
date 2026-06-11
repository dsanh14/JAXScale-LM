"""Generate text from a trained checkpoint.

Examples:
    uv run python scripts/generate.py \
        --checkpoint artifacts/checkpoints/cpu_smoke/latest \
        --prompt "Once upon a time" --max-new-tokens 64 --use-kv-cache

    uv run python scripts/generate.py \
        --checkpoint artifacts/checkpoints/cpu_smoke/latest \
        --prompt "the cat" --do-sample --temperature 0.8 --top-k 50 --seed 7
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir, /latest, or /<step>")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--use-kv-cache", dest="use_kv_cache", action="store_true")
    cache_group.add_argument("--no-kv-cache", dest="use_kv_cache", action="store_false")
    parser.set_defaults(use_kv_cache=True)
    args = parser.parse_args()

    from jaxscale_lm.config import InferenceConfig
    from jaxscale_lm.inference.engine import InferenceEngine
    from jaxscale_lm.utils.logging import setup_logging

    setup_logging()
    engine = InferenceEngine.from_checkpoint(args.checkpoint)
    options = InferenceConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        seed=args.seed,
        use_kv_cache=args.use_kv_cache,
    )
    result = engine.generate(args.prompt, options)

    print(f"prompt:            {args.prompt!r}")
    print(f"completion:        {result.generated_text!r}")
    print(f"prompt tokens:     {result.prompt_tokens}")
    print(f"generated tokens:  {result.generated_tokens}")
    print(f"kv cache:          {result.cache_enabled}")
    print(
        f"prefill latency:   {result.timing.prefill_s * 1000:.1f} ms (includes first-call compile)"
    )
    print(f"decode latency:    {result.timing.decode_s * 1000:.1f} ms")
    print(f"tokens/second:     {result.tokens_per_second:.1f}")


if __name__ == "__main__":
    main()
