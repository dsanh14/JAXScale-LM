"""Token-weighted metric aggregation across batches (and devices).

Perplexity is **never** averaged per batch: the aggregator accumulates total
NLL and total valid tokens, and exponentiates the final token-weighted mean
loss once. With unequal batch sizes (the final eval batch may be short) a
mean-of-means would be biased.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jaxscale_lm.training.loss import LossStats, perplexity


@dataclass
class MetricAggregator:
    """Accumulates LossStats sums; produces token-weighted summary metrics."""

    total_nll: float = 0.0
    total_correct: float = 0.0
    total_tokens: float = 0.0
    batches: int = field(default=0)

    def update(self, stats: LossStats) -> None:
        self.total_nll += float(stats.nll_sum)
        self.total_correct += float(stats.correct_sum)
        self.total_tokens += float(stats.valid_tokens)
        self.batches += 1

    def summary(self) -> dict[str, float]:
        if self.total_tokens == 0:
            raise ValueError(
                "No valid tokens were aggregated; evaluation saw no data. "
                "Check evaluation.num_batches and the validation split size."
            )
        mean_nll = self.total_nll / self.total_tokens
        return {
            "loss": mean_nll,
            "perplexity": perplexity(mean_nll),
            "accuracy": self.total_correct / self.total_tokens,
            "valid_tokens": self.total_tokens,
            "batches": float(self.batches),
        }
