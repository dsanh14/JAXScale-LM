"""Prometheus metrics for the serving layer.

Label discipline: only low-cardinality labels (model_id, cache flag, error
kind). Never prompts, request ids, or user-supplied strings.
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Module-level registry so tests can create isolated apps.
registry = CollectorRegistry()

REQUESTS = Counter(
    "jaxscale_requests_total",
    "Generation requests received",
    ["cache_enabled"],
    registry=registry,
)
REQUEST_ERRORS = Counter(
    "jaxscale_request_errors_total",
    "Failed generation requests",
    ["kind"],  # "validation" | "internal" | "no_model"
    registry=registry,
)
ACTIVE_REQUESTS = Gauge(
    "jaxscale_active_requests",
    "Generation requests currently executing",
    registry=registry,
)
REQUEST_LATENCY = Histogram(
    "jaxscale_request_latency_seconds",
    "End-to-end generation request latency",
    registry=registry,
)
PREFILL_LATENCY = Histogram(
    "jaxscale_prefill_latency_seconds",
    "Prompt prefill latency",
    registry=registry,
)
DECODE_LATENCY = Histogram(
    "jaxscale_decode_latency_seconds",
    "Decode-loop latency",
    registry=registry,
)
GENERATED_TOKENS = Counter(
    "jaxscale_generated_tokens_total",
    "Tokens generated",
    registry=registry,
)
PROMPT_TOKENS = Counter(
    "jaxscale_prompt_tokens_total",
    "Prompt tokens processed",
    registry=registry,
)
TOKENS_PER_SECOND = Gauge(
    "jaxscale_last_tokens_per_second",
    "Decode tokens/second of the most recent request",
    registry=registry,
)
MODEL_LOAD_SECONDS = Histogram(
    "jaxscale_model_load_seconds",
    "Checkpoint load duration",
    registry=registry,
)
MODEL_LOAD_FAILURES = Counter(
    "jaxscale_model_load_failures_total",
    "Model load failures",
    registry=registry,
)
WARMUP_SECONDS = Gauge(
    "jaxscale_warmup_seconds",
    "Most recent compilation warmup duration",
    registry=registry,
)


def render() -> bytes:
    """Render the metrics exposition document."""
    return generate_latest(registry)
