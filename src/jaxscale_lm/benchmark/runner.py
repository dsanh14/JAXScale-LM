"""Benchmark runner: executes suites, persists JSONL/CSV/Markdown/plots.

Output layout (under ``<benchmark_dir>/<run_id>/``):

- ``records.jsonl``   — every record, raw samples included (append-as-you-go)
- ``summary.csv``     — flat table of key columns
- ``summary.md``      — human-readable summary grouped by suite
- ``resolved_config.yaml`` — exact configuration of the run
- ``plots/*.png``     — standard plot set

Suite failures are caught at the suite boundary, recorded as failed records,
and re-raised only if *every* suite failed (a fully broken environment).
"""

from __future__ import annotations

import csv
import json
import traceback
from pathlib import Path

from jaxscale_lm.benchmark import compilation, inference, training
from jaxscale_lm.benchmark.plots import render_all
from jaxscale_lm.benchmark.schema import (
    BenchmarkRecord,
    environment_info,
    new_run_id,
)
from jaxscale_lm.config import Config, save_resolved_config
from jaxscale_lm.utils.logging import get_logger, log_event

_logger = get_logger("benchmark")

_CSV_COLUMNS = [
    "suite",
    "name",
    "mode",
    "status",
    "batch_size",
    "sequence_length",
    "prompt_length",
    "generate_length",
    "dtype",
    "measure_iterations",
    "mean_s",
    "median_s",
    "std_s",
    "p50_s",
    "p90_s",
    "p95_s",
    "p99_s",
]


class BenchmarkRunner:
    """Executes the configured suites and writes all artifacts."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.run_id = new_run_id()
        self.out_dir = Path(config.benchmark_dir) / self.run_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.environment = environment_info()
        self.records: list[BenchmarkRecord] = []
        save_resolved_config(config, self.out_dir / "resolved_config.yaml")

    # -- record handling --------------------------------------------------
    def _finalize(self, record: BenchmarkRecord) -> None:
        record.run_id = self.run_id
        record.environment = self.environment
        record.model_config = self.config.model.model_dump(mode="json")
        from datetime import UTC, datetime

        record.timestamp = datetime.now(tz=UTC).isoformat()
        self.records.append(record)
        with (self.out_dir / "records.jsonl").open("a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    # -- execution ----------------------------------------------------------
    def run(self) -> Path:
        suites = set(self.config.benchmark.suites)
        log_event(
            _logger,
            "benchmark run starting",
            run_id=self.run_id,
            suites=sorted(suites),
            platform=self.environment["platform"],
            devices=self.environment["device_count"],
        )

        suite_errors: list[str] = []
        plans: list[tuple[str, object]] = []
        if "compilation" in suites:
            plans.append(("compilation", lambda: compilation.run(self.config)))
        if "training" in suites:
            plans.append(("training", lambda: training.run(self.config)))
        inference_suites = suites & {"prefill", "decode", "e2e", "cache"}
        if inference_suites:
            plans.append(("inference", lambda: inference.run(self.config, inference_suites)))

        for suite_name, plan in plans:
            try:
                for record in plan():  # type: ignore[operator]
                    self._finalize(record)
                log_event(_logger, "suite complete", suite=suite_name)
            except Exception as exc:
                # Record the failure (spec: failed runs are recorded, not
                # dropped) and continue with the remaining suites.
                trace = traceback.format_exc()
                _logger.error("suite failed", extra={"suite": suite_name, "error": str(exc)})
                self._finalize(
                    BenchmarkRecord(
                        suite=suite_name,
                        name="suite_execution",
                        mode="n/a",
                        status="failed",
                        error=f"{exc}\n{trace}",
                    )
                )
                suite_errors.append(suite_name)

        if suite_errors and len(suite_errors) == len(plans):
            raise RuntimeError(
                f"All benchmark suites failed ({suite_errors}); see "
                f"{self.out_dir / 'records.jsonl'} for tracebacks."
            )

        self._write_csv()
        self._write_markdown()
        render_all(self.records, self.out_dir / "plots")
        log_event(_logger, "benchmark run complete", output=str(self.out_dir))
        return self.out_dir

    # -- outputs -----------------------------------------------------------
    def _write_csv(self) -> None:
        with (self.out_dir / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for record in self.records:
                writer.writerow(record.to_dict())

    def _write_markdown(self) -> None:
        env = self.environment
        lines = [
            f"# Benchmark run `{self.run_id}`",
            "",
            f"- platform: **{env['platform']}** ({', '.join(env['device_names'])}), "
            f"{env['device_count']} device(s)",
            f"- host: {env['host']}",
            f"- git: `{env['git_commit'] or 'unknown'}`"
            + (" (dirty)" if env.get("git_dirty") else ""),
            f"- versions: jax {env['jax_version']}, flax {env['flax_version']}, "
            f"optax {env['optax_version']}, orbax {env['orbax_version']}, "
            f"python {env['python_version']}",
            f"- model: {self.config.model.num_layers}L x {self.config.model.hidden_size}h, "
            f"vocab {self.config.model.vocab_size}",
            "",
        ]
        for suite in sorted({r.suite for r in self.records}):
            suite_records = [r for r in self.records if r.suite == suite]
            lines += [f"## {suite}", ""]
            lines.append("| name | mode | status | median | p90 | std | iters | notes |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for r in suite_records:
                med = f"{r.median_s * 1000:.2f} ms" if r.median_s is not None else "-"
                p90 = f"{r.p90_s * 1000:.2f} ms" if r.p90_s is not None else "-"
                std = f"{r.std_s * 1000:.2f}" if r.std_s is not None else "-"
                notes = []
                for key in ("tokens_per_second", "ms_per_token", "first_call_over_steady"):
                    if key in r.extra:
                        value = r.extra[key]
                        notes.append(
                            f"{key}={value:.1f}" if isinstance(value, float) else str(value)
                        )
                if r.status == "failed":
                    notes.append(f"ERROR: {(r.error or '')[:120]}")
                lines.append(
                    f"| {r.name} | {r.mode} | {r.status} | {med} | {p90} | {std} "
                    f"| {r.measure_iterations or '-'} | {'; '.join(notes)} |"
                )
            lines.append("")
        (self.out_dir / "summary.md").write_text("\n".join(lines))
