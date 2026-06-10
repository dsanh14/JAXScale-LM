"""Inference tests: sampling filters, generation loops, cached/naive equivalence."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jaxscale_lm.config import ModelConfig
from jaxscale_lm.inference.decode import make_cached_decode_fn, make_naive_decode_fn
from jaxscale_lm.inference.generate import cached_generate, naive_generate
from jaxscale_lm.inference.prefill import make_prefill_fn
from jaxscale_lm.inference.sampling import (
    SamplingParams,
    apply_repetition_penalty,
    select_token,
    top_k_filter,
    top_p_filter,
)
from jaxscale_lm.model.cache import init_cache
from jaxscale_lm.model.transformer import build_model
from jaxscale_lm.utils.seed import make_key

pytestmark = pytest.mark.unit

CFG = ModelConfig(
    vocab_size=61,
    max_sequence_length=48,
    num_layers=2,
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=4,
)


@pytest.fixture(scope="module")
def setup():
    model = build_model(CFG, seed=0)
    graphdef, params = nnx.split(model)
    return {
        "params": params,
        "prefill": jax.jit(make_prefill_fn(graphdef)),
        "decode": jax.jit(make_cached_decode_fn(graphdef)),
        "naive": jax.jit(make_naive_decode_fn(graphdef)),
    }


def _prompt(batch: int = 1, length: int = 8, seed: int = 0) -> jax.Array:
    return jax.random.randint(jax.random.key(seed), (batch, length), 0, CFG.vocab_size)


class TestTopK:
    def test_keeps_exactly_k(self):
        logits = jnp.asarray([[1.0, 5.0, 3.0, 2.0, 4.0]])
        filtered = top_k_filter(logits, 2)
        kept = np.asarray(filtered[0]) > -1e29
        assert kept.tolist() == [False, True, False, False, True]

    def test_zero_disables(self):
        logits = jnp.asarray([[1.0, 2.0]])
        np.testing.assert_array_equal(np.asarray(top_k_filter(logits, 0)), np.asarray(logits))


class TestTopP:
    def test_keeps_nucleus(self):
        # probs ~ [0.643, 0.236, 0.087, 0.032] -> p=0.7 keeps tokens 0 and 1.
        logits = jnp.log(jnp.asarray([[0.643, 0.236, 0.087, 0.032]]))
        filtered = top_p_filter(logits, 0.7)
        kept = np.asarray(filtered[0]) > -1e29
        assert kept.tolist() == [True, True, False, False]

    def test_always_keeps_top_token(self):
        logits = jnp.asarray([[10.0, 0.0, -5.0]])
        filtered = top_p_filter(logits, 0.01)
        kept = np.asarray(filtered[0]) > -1e29
        assert kept.tolist() == [True, False, False]

    def test_one_disables(self):
        logits = jnp.asarray([[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(np.asarray(top_p_filter(logits, 1.0)), np.asarray(logits))


class TestRepetitionPenalty:
    def test_penalizes_seen_tokens_only(self):
        logits = jnp.asarray([[2.0, -2.0, 1.0]])
        seen = jnp.asarray([[True, True, False]])
        out = np.asarray(apply_repetition_penalty(logits, seen, 2.0)[0])
        assert out[0] == pytest.approx(1.0)  # positive: divided
        assert out[1] == pytest.approx(-4.0)  # negative: multiplied
        assert out[2] == pytest.approx(1.0)  # unseen: untouched


class TestSelectToken:
    def test_greedy_is_argmax(self):
        logits = jnp.asarray([[0.1, 0.9, 0.5], [2.0, 0.0, 1.0]])
        token = select_token(logits, SamplingParams(do_sample=False), make_key(0))
        assert token.tolist() == [1, 0]

    def test_sampling_deterministic_under_key(self):
        logits = jax.random.normal(jax.random.key(3), (2, 16))
        params = SamplingParams(do_sample=True, temperature=1.0, top_k=8)
        a = select_token(logits, params, make_key(7))
        b = select_token(logits, params, make_key(7))
        assert a.tolist() == b.tolist()

    def test_validation(self):
        with pytest.raises(ValueError, match="temperature"):
            SamplingParams(do_sample=True, temperature=0.0).validate(100)
        with pytest.raises(ValueError, match="top_k"):
            SamplingParams(top_k=200).validate(100)
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=0.0).validate(100)


class TestGenerationLoops:
    def test_greedy_cached_deterministic(self, setup):
        prompt = _prompt()
        outs = []
        for _ in range(2):
            cache = init_cache(CFG, batch_size=1, capacity=CFG.max_sequence_length)
            out = cached_generate(
                setup["prefill"],
                setup["decode"],
                setup["params"],
                prompt,
                cache,
                max_new_tokens=12,
                sampling=SamplingParams(),
                key=make_key(0),
                eos_id=None,
                pad_id=0,
                vocab_size=CFG.vocab_size,
            )
            outs.append(out.token_ids)
        np.testing.assert_array_equal(outs[0], outs[1])

    def test_cached_equals_naive_greedy(self, setup):
        """The KV-cache path and full-recompute path must produce identical
        greedy outputs (they compute the same math)."""
        prompt = _prompt(batch=2, length=6)
        cache = init_cache(CFG, batch_size=2, capacity=CFG.max_sequence_length)
        cached = cached_generate(
            setup["prefill"],
            setup["decode"],
            setup["params"],
            prompt,
            cache,
            max_new_tokens=10,
            sampling=SamplingParams(),
            key=make_key(0),
            eos_id=None,
            pad_id=0,
            vocab_size=CFG.vocab_size,
        )
        naive = naive_generate(
            setup["naive"],
            setup["params"],
            prompt,
            capacity=CFG.max_sequence_length,
            max_new_tokens=10,
            sampling=SamplingParams(),
            key=make_key(0),
            eos_id=None,
            pad_id=0,
            vocab_size=CFG.vocab_size,
        )
        np.testing.assert_array_equal(cached.token_ids, naive.token_ids)

    def test_eos_stops_generation(self, setup):
        """Force EOS by treating the first greedily-chosen token as EOS."""
        prompt = _prompt()
        cache = init_cache(CFG, batch_size=1, capacity=CFG.max_sequence_length)
        first = cached_generate(
            setup["prefill"],
            setup["decode"],
            setup["params"],
            prompt,
            cache,
            max_new_tokens=1,
            sampling=SamplingParams(),
            key=make_key(0),
            eos_id=None,
            pad_id=0,
            vocab_size=CFG.vocab_size,
        )
        eos = int(first.token_ids[0, 0])
        cache = init_cache(CFG, batch_size=1, capacity=CFG.max_sequence_length)
        out = cached_generate(
            setup["prefill"],
            setup["decode"],
            setup["params"],
            prompt,
            cache,
            max_new_tokens=20,
            sampling=SamplingParams(),
            key=make_key(0),
            eos_id=eos,
            pad_id=0,
            vocab_size=CFG.vocab_size,
        )
        assert out.timing.decode_steps == 1  # stopped immediately on EOS
        assert int(out.token_ids[0, 0]) == eos

    def test_batch_dimension_preserved(self, setup):
        prompt = _prompt(batch=3, length=5)
        cache = init_cache(CFG, batch_size=3, capacity=CFG.max_sequence_length)
        out = cached_generate(
            setup["prefill"],
            setup["decode"],
            setup["params"],
            prompt,
            cache,
            max_new_tokens=4,
            sampling=SamplingParams(),
            key=make_key(0),
            eos_id=None,
            pad_id=0,
            vocab_size=CFG.vocab_size,
        )
        assert out.token_ids.shape == (3, 4)

    def test_overflow_rejected(self, setup):
        prompt = _prompt(length=10)
        with pytest.raises(ValueError, match="capacity"):
            naive_generate(
                setup["naive"],
                setup["params"],
                prompt,
                capacity=12,
                max_new_tokens=10,
                sampling=SamplingParams(),
                key=make_key(0),
                eos_id=None,
                pad_id=0,
                vocab_size=CFG.vocab_size,
            )
