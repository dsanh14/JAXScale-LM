"""Offline Matplotlib plots from benchmark records.

Each plot is generated only when the records it needs exist; missing data is
logged and skipped, never fabricated. Error bars show +-1 std where more
than one sample exists.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt

from jaxscale_lm.benchmark.schema import BenchmarkRecord
from jaxscale_lm.utils.logging import get_logger, log_event

_logger = get_logger("plots")


def _ok(records: list[BenchmarkRecord], suite: str, mode: str | None = None):
    return [
        r
        for r in records
        if r.suite == suite and r.status == "ok" and (mode is None or r.mode == mode)
    ]


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log_event(_logger, "plot written", path=str(path))
    return path


def plot_compile_vs_seq(records: list[BenchmarkRecord], out_dir: Path) -> Path | None:
    first = _ok(records, "compilation", "first_call")
    steady = _ok(records, "compilation", "steady_state")
    if not first or not steady:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, recs in (("first call (compile+run)", first), ("steady state", steady)):
        recs = sorted(recs, key=lambda r: r.sequence_length or 0)
        xs = [r.sequence_length or 0 for r in recs]
        ys = [(r.median_s or 0) * 1000 for r in recs]
        errs = [(r.std_s or 0) * 1000 for r in recs]
        ax.errorbar(xs, ys, yerr=errs, marker="o", label=label, capsize=3)
    ax.set_xlabel("sequence length")
    ax.set_ylabel("latency (ms)")
    ax.set_yscale("log")
    ax.set_title("Forward pass: compile cost vs steady state")
    ax.legend()
    return _save(fig, out_dir / "compile_time_vs_sequence_length.png")


def plot_train_throughput(records: list[BenchmarkRecord], out_dir: Path) -> Path | None:
    recs = [r for r in _ok(records, "training") if "tokens_per_second" in r.extra]
    recs = [r for r in recs if r.extra.get("accumulation_steps", 1) == 1]
    if not recs:
        return None
    by_dtype: dict[str, list[BenchmarkRecord]] = {}
    for r in recs:
        by_dtype.setdefault(r.dtype or "?", []).append(r)
    fig, ax = plt.subplots(figsize=(6, 4))
    for dtype, group in by_dtype.items():
        group = sorted(group, key=lambda r: r.batch_size or 0)
        ax.plot(
            [r.batch_size or 0 for r in group],
            [r.extra["tokens_per_second"] for r in group],
            marker="o",
            label=dtype,
        )
    ax.set_xlabel("batch size")
    ax.set_ylabel("tokens / second")
    ax.set_title("Training throughput vs batch size")
    ax.legend()
    return _save(fig, out_dir / "training_throughput_vs_batch_size.png")


def plot_prefill_latency(records: list[BenchmarkRecord], out_dir: Path) -> Path | None:
    recs = _ok(records, "prefill")
    if not recs:
        return None
    by_batch: dict[int, list[BenchmarkRecord]] = {}
    for r in recs:
        by_batch.setdefault(r.batch_size or 0, []).append(r)
    fig, ax = plt.subplots(figsize=(6, 4))
    for batch, group in sorted(by_batch.items()):
        group = sorted(group, key=lambda r: r.prompt_length or 0)
        ax.errorbar(
            [r.prompt_length or 0 for r in group],
            [(r.median_s or 0) * 1000 for r in group],
            yerr=[(r.std_s or 0) * 1000 for r in group],
            marker="o",
            capsize=3,
            label=f"batch={batch}",
        )
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("prefill latency (ms)")
    ax.set_title("Prefill latency vs prompt length")
    ax.legend()
    return _save(fig, out_dir / "prefill_latency_vs_prompt_length.png")


def plot_decode_latency(records: list[BenchmarkRecord], out_dir: Path) -> Path | None:
    recs = _ok(records, "decode")
    if not recs:
        return None
    by_ctx: dict[int, list[BenchmarkRecord]] = {}
    for r in recs:
        by_ctx.setdefault(r.prompt_length or 0, []).append(r)
    fig, ax = plt.subplots(figsize=(6, 4))
    for ctx, group in sorted(by_ctx.items()):
        group = sorted(group, key=lambda r: r.batch_size or 0)
        ax.errorbar(
            [r.batch_size or 0 for r in group],
            [(r.median_s or 0) * 1000 for r in group],
            yerr=[(r.std_s or 0) * 1000 for r in group],
            marker="o",
            capsize=3,
            label=f"context={ctx}",
        )
    ax.set_xlabel("batch size")
    ax.set_ylabel("decode step latency (ms)")
    ax.set_title("Decode latency vs batch size")
    ax.legend()
    return _save(fig, out_dir / "decode_latency_vs_batch_size.png")


def plot_cache_comparison(records: list[BenchmarkRecord], out_dir: Path) -> Path | None:
    recs = _ok(records, "cache")
    cached = [r for r in recs if r.extra.get("kv_cache") is True]
    naive = [r for r in recs if r.extra.get("kv_cache") is False]
    if not cached or not naive:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    gens = sorted({r.generate_length or 0 for r in cached})
    for offset, (label, group) in enumerate((("KV-cached", cached), ("naive", naive))):
        ys = []
        for g in gens:
            match = [r for r in group if r.generate_length == g]
            ys.append((match[0].median_s or 0) * 1000 if match else 0)
        ax.bar(
            [i + offset * width for i in range(len(gens))],
            ys,
            width,
            label=label,
        )
    ax.set_xticks([i + width / 2 for i in range(len(gens))])
    ax.set_xticklabels([str(g) for g in gens])
    ax.set_xlabel("generated tokens")
    ax.set_ylabel("decode time (ms)")
    ax.set_title("Naive full-prefix vs KV-cached decoding")
    ax.legend()
    return _save(fig, out_dir / "naive_vs_cached_generation.png")


def plot_tokens_per_second(records: list[BenchmarkRecord], out_dir: Path) -> Path | None:
    recs = [r for r in _ok(records, "e2e") if "generated_tokens_per_second" in r.extra]
    if not recs:
        return None
    by_prompt: dict[int, list[BenchmarkRecord]] = {}
    for r in recs:
        by_prompt.setdefault(r.prompt_length or 0, []).append(r)
    fig, ax = plt.subplots(figsize=(6, 4))
    for prompt, group in sorted(by_prompt.items()):
        group = sorted(group, key=lambda r: r.generate_length or 0)
        ax.plot(
            [r.generate_length or 0 for r in group],
            [r.extra["generated_tokens_per_second"] for r in group],
            marker="o",
            label=f"prompt={prompt}",
        )
    ax.set_xlabel("generated tokens")
    ax.set_ylabel("decode tokens / second")
    ax.set_title("Generation throughput vs generated length")
    ax.legend()
    return _save(fig, out_dir / "tokens_per_second_vs_generated_length.png")


def render_all(records: list[BenchmarkRecord], out_dir: Path) -> list[Path]:
    """Render every plot whose data exists; returns written paths."""
    out: list[Path] = []
    for fn in (
        plot_compile_vs_seq,
        plot_train_throughput,
        plot_prefill_latency,
        plot_decode_latency,
        plot_cache_comparison,
        plot_tokens_per_second,
    ):
        path = fn(records, out_dir)
        if path is None:
            log_event(_logger, "plot skipped (no data)", plot=fn.__name__)
        else:
            out.append(path)
    return out
