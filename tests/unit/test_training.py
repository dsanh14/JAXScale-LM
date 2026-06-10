"""Training-system unit and numerical tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxscale_lm.config import ModelConfig, OptimizerConfig
from jaxscale_lm.model.transformer import build_model
from jaxscale_lm.training.loss import LossStats, loss_stats, mean_loss
from jaxscale_lm.training.metrics import MetricAggregator
from jaxscale_lm.training.optimizer import build_optimizer, build_schedule, weight_decay_mask
from jaxscale_lm.training.state import create_train_state
from jaxscale_lm.training.step import make_eval_step, make_train_step
from jaxscale_lm.types import Batch
from jaxscale_lm.utils.seed import make_key

pytestmark = pytest.mark.unit

CFG = ModelConfig(
    vocab_size=61,
    max_sequence_length=32,
    num_layers=2,
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=4,
)


def _batch(accum: int, micro: int, seq: int, seed: int = 0) -> Batch:
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, CFG.vocab_size, size=(accum, micro, seq + 1)).astype(np.int32)
    return Batch(
        input_ids=jnp.asarray(ids[..., :-1]),
        target_ids=jnp.asarray(ids[..., 1:]),
        loss_mask=jnp.ones((accum, micro, seq), jnp.float32),
    )


def _setup(accum: int = 1, max_steps: int = 100, opt_cfg: OptimizerConfig | None = None):
    model = build_model(CFG, seed=0)
    tx, schedule = build_optimizer(opt_cfg or OptimizerConfig(warmup_steps=0), max_steps)
    graphdef, state = create_train_state(model, tx, make_key(0))
    step_fn = jax.jit(make_train_step(graphdef, tx, schedule, accum))
    return graphdef, state, step_fn


class TestLoss:
    def test_perfect_prediction_low_loss(self):
        # Logits hugely favoring the target -> NLL ~ 0, accuracy 1.
        targets = jnp.asarray([[1, 2, 3]], jnp.int32)
        logits = jax.nn.one_hot(targets, 8) * 100.0
        stats = loss_stats(logits, targets, jnp.ones((1, 3)))
        assert float(stats.nll_sum) == pytest.approx(0.0, abs=1e-4)
        assert float(stats.correct_sum) == 3.0
        assert float(stats.valid_tokens) == 3.0

    def test_uniform_logits_log_vocab(self):
        vocab = 16
        logits = jnp.zeros((1, 4, vocab))
        targets = jnp.zeros((1, 4), jnp.int32)
        stats = loss_stats(logits, targets, jnp.ones((1, 4)))
        assert float(mean_loss(stats)) == pytest.approx(np.log(vocab), rel=1e-5)

    def test_mask_excludes_tokens(self):
        logits = jnp.zeros((1, 4, 8))
        targets = jnp.zeros((1, 4), jnp.int32)
        mask = jnp.asarray([[1.0, 1.0, 0.0, 0.0]])
        stats = loss_stats(logits, targets, mask)
        assert float(stats.valid_tokens) == 2.0
        assert float(stats.nll_sum) == pytest.approx(2 * np.log(8), rel=1e-5)


class TestMetricAggregation:
    def test_token_weighted_not_batch_averaged(self):
        agg = MetricAggregator()
        # Batch A: 1 token, loss 4.0 | Batch B: 9 tokens, loss 2.0.
        agg.update(LossStats(jnp.asarray(4.0), jnp.asarray(0.0), jnp.asarray(1.0)))
        agg.update(LossStats(jnp.asarray(18.0), jnp.asarray(9.0), jnp.asarray(9.0)))
        summary = agg.summary()
        # Token-weighted mean: 22/10 = 2.2 (NOT (4+2)/2 = 3.0).
        assert summary["loss"] == pytest.approx(2.2)
        assert summary["perplexity"] == pytest.approx(np.exp(2.2), rel=1e-6)
        assert summary["accuracy"] == pytest.approx(0.9)

    def test_empty_aggregation_rejected(self):
        with pytest.raises(ValueError, match="No valid tokens"):
            MetricAggregator().summary()


class TestSchedule:
    def test_warmup_then_cosine(self):
        cfg = OptimizerConfig(learning_rate=1e-3, warmup_steps=10, schedule="cosine")
        schedule = build_schedule(cfg, max_steps=100)
        assert float(schedule(0)) == pytest.approx(0.0)
        assert float(schedule(10)) == pytest.approx(1e-3, rel=1e-3)
        assert float(schedule(100)) == pytest.approx(1e-4, rel=1e-2)  # min ratio 0.1

    def test_warmup_exceeding_max_steps_rejected(self):
        with pytest.raises(ValueError, match="warmup_steps"):
            build_schedule(OptimizerConfig(warmup_steps=200), max_steps=100)

    def test_constant_schedule(self):
        cfg = OptimizerConfig(learning_rate=1e-3, warmup_steps=0, schedule="constant")
        schedule = build_schedule(cfg, max_steps=100)
        assert float(schedule(50)) == pytest.approx(1e-3)


class TestWeightDecayMask:
    def test_only_matrices_decayed(self):
        tree = {"w": jnp.zeros((4, 4)), "scale": jnp.zeros((4,)), "b": jnp.zeros(())}
        mask = weight_decay_mask(tree)
        assert mask["w"] is True
        assert mask["scale"] is False
        assert mask["b"] is False


class TestTrainStep:
    def test_loss_finite_and_params_change(self):
        _, state, step_fn = _setup()
        before = jax.tree.leaves(state.params)
        state, metrics = step_fn(state, _batch(1, 4, 16))
        assert np.isfinite(float(metrics["loss"]))
        assert int(state.step) == 1
        after = jax.tree.leaves(state.params)
        changed = any(
            not np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(before, after, strict=True)
        )
        assert changed, "optimizer step did not modify parameters"

    def test_loss_decreases_on_repeated_batch(self):
        _, state, step_fn = _setup()
        batch = _batch(1, 4, 16)
        first = None
        for _ in range(20):
            state, metrics = step_fn(state, batch)
            if first is None:
                first = float(metrics["loss"])
        assert float(metrics["loss"]) < first

    def test_deterministic_given_seed(self):
        _, state_a, step_fn = _setup()
        _, state_b, _ = _setup()
        batch = _batch(1, 4, 16)
        a, metrics_a = step_fn(state_a, batch)
        b, metrics_b = step_fn(state_b, batch)
        assert float(metrics_a["loss"]) == float(metrics_b["loss"])
        for x, y in zip(jax.tree.leaves(a.params), jax.tree.leaves(b.params), strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


class TestGradientAccumulation:
    def test_accumulated_matches_large_batch(self):
        """2 microbatches of 4 accumulated == 1 batch of 8, within tolerance.

        Tolerance: float32 summation order differs between the two paths;
        1e-5 relative on parameters is far below any meaningful drift.
        """
        big = _batch(1, 8, 16, seed=7)
        split = Batch(*(x.reshape(2, 4, *x.shape[2:]) for x in big))

        _, state_big, step_big = _setup(accum=1)
        _, state_acc, step_acc = _setup(accum=2)

        new_big, metrics_big = step_big(state_big, big)
        new_acc, metrics_acc = step_acc(state_acc, split)

        assert float(metrics_big["loss"]) == pytest.approx(float(metrics_acc["loss"]), rel=1e-6)
        for a, b in zip(
            jax.tree.leaves(new_big.params), jax.tree.leaves(new_acc.params), strict=True
        ):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-7)


class TestEvalStep:
    def test_eval_does_not_mutate_params(self):
        graphdef, state, _ = _setup()
        eval_fn = jax.jit(make_eval_step(graphdef))
        before = [np.asarray(x).copy() for x in jax.tree.leaves(state.params)]
        batch = _batch(1, 4, 16)
        stats = eval_fn(state.params, Batch(*(x[0] for x in batch)))
        assert np.isfinite(float(stats.nll_sum))
        for x, y in zip(before, jax.tree.leaves(state.params), strict=True):
            np.testing.assert_array_equal(x, np.asarray(y))

    def test_eval_deterministic_with_dropout_config(self):
        cfg = CFG.model_copy(update={"dropout_rate": 0.5, "attention_dropout_rate": 0.5})
        model = build_model(cfg, seed=0)
        tx, schedule = build_optimizer(OptimizerConfig(warmup_steps=0), 100)
        graphdef, state = create_train_state(model, tx, make_key(0))
        eval_fn = jax.jit(make_eval_step(graphdef))
        batch = _batch(1, 4, 16)
        squeezed = Batch(*(x[0] for x in batch))
        a = eval_fn(state.params, squeezed)
        b = eval_fn(state.params, squeezed)
        assert float(a.nll_sum) == float(b.nll_sum)
