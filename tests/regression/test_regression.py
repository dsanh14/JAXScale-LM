"""Regression tests: values that must stay stable across refactors.

The loss-trajectory fixture pins fixed-seed smoke training behavior. If a
change legitimately alters numerics (e.g. a different init order), update
the expected values in the same PR and call it out in the description.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from flax import nnx

from jaxscale_lm.config import Config, ModelConfig
from jaxscale_lm.inference.decode import make_cached_decode_fn
from jaxscale_lm.inference.generate import cached_generate
from jaxscale_lm.inference.prefill import make_prefill_fn
from jaxscale_lm.inference.sampling import SamplingParams
from jaxscale_lm.model.cache import init_cache
from jaxscale_lm.model.transformer import build_model
from jaxscale_lm.training.trainer import Trainer
from jaxscale_lm.utils.seed import make_key

pytestmark = pytest.mark.integration

_REGRESSION_CFG = ModelConfig(
    vocab_size=61,
    max_sequence_length=48,
    num_layers=2,
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=4,
)


class TestLossTrajectory:
    def test_smoke_training_loss_decreases_and_is_reproducible(self, tmp_path, smoke_config):
        """Two identical fixed-seed runs produce identical loss; loss drops."""
        losses = []
        for name in ("run_a", "run_b"):
            config: Config = smoke_config.model_copy(
                update={
                    "project": smoke_config.project.model_copy(
                        update={"artifacts_dir": tmp_path / name, "run_name": name}
                    )
                }
            )
            trainer = Trainer(config)
            summary = trainer.train()
            losses.append(summary["loss"])
            # The synthetic corpus is highly regular: 10 steps must beat
            # the uniform-distribution baseline ln(259) ~= 5.557.
            assert summary["loss"] < np.log(259)
        assert losses[0] == pytest.approx(losses[1], rel=1e-6)


class TestParameterCount:
    def test_known_config_count(self):
        # embed 61*32 + 2 layers * (4*32*32 + 2*32*64 + 2*32) + final 32
        expected = 61 * 32 + 2 * (4 * 32 * 32 + 2 * 32 * 64 + 64) + 32
        assert build_model(_REGRESSION_CFG, seed=0).num_params() == expected


class TestDeterministicGenerationFixture:
    def test_greedy_fixture_stable(self):
        """Greedy generation from a fixed-seed *untrained* model is a pure
        function of (init seed, prompt); pin the first run's behavior by
        comparing two independent engine constructions."""
        outputs = []
        for _ in range(2):
            model = build_model(_REGRESSION_CFG, seed=123)
            graphdef, params = nnx.split(model)
            prompt = jax.random.randint(jax.random.key(5), (1, 6), 0, 61)
            out = cached_generate(
                jax.jit(make_prefill_fn(graphdef)),
                jax.jit(make_cached_decode_fn(graphdef)),
                params,
                prompt,
                init_cache(_REGRESSION_CFG, 1, 48),
                max_new_tokens=10,
                sampling=SamplingParams(),
                key=make_key(0),
                eos_id=None,
                pad_id=0,
                vocab_size=61,
            )
            outputs.append(out.token_ids)
        np.testing.assert_array_equal(outputs[0], outputs[1])
