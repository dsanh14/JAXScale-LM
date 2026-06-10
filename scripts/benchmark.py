"""Run the benchmark harness.

Examples:
    uv run python scripts/benchmark.py --config configs/benchmark/default.yaml
    uv run python scripts/benchmark.py --config configs/benchmark/default.yaml --quick
    uv run python scripts/benchmark.py --config configs/benchmark/default.yaml \
        --suites compilation cache
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument(
        "--quick", action="store_true", help="Fewer iterations/sweep points for a fast pass"
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=None,
        help="Subset of suites (compilation training prefill decode e2e cache)",
    )
    args = parser.parse_args()

    from jaxscale_lm.benchmark.runner import BenchmarkRunner
    from jaxscale_lm.config import load_config
    from jaxscale_lm.utils.logging import setup_logging

    config = load_config(args.config)
    setup_logging(config.logging.level, config.logging.json_format)

    bench_updates = {}
    if args.quick:
        bench_updates.update(
            {
                "warmup_iterations": 1,
                "measure_iterations": 3,
                "batch_sizes": tuple(config.benchmark.batch_sizes[:2]),
                "prompt_lengths": tuple(config.benchmark.prompt_lengths[:2]),
                "sequence_lengths": tuple(config.benchmark.sequence_lengths[:2]),
            }
        )
    if args.suites:
        bench_updates["suites"] = tuple(args.suites)
    if bench_updates:
        config = config.model_copy(
            update={"benchmark": config.benchmark.model_copy(update=bench_updates)}
        )

    runner = BenchmarkRunner(config)
    out_dir = runner.run()
    ok = sum(1 for r in runner.records if r.status == "ok")
    failed = sum(1 for r in runner.records if r.status == "failed")
    print(f"records: {ok} ok, {failed} failed")
    print(f"outputs: {out_dir}")
    print(f"  - {out_dir / 'records.jsonl'}")
    print(f"  - {out_dir / 'summary.csv'}")
    print(f"  - {out_dir / 'summary.md'}")
    print(f"  - {out_dir / 'plots/'}")


if __name__ == "__main__":
    main()
