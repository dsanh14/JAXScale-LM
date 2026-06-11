"""Model tests: shapes, causality, RoPE, KV cache, parameter counting."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jaxscale_lm.config import ModelConfig
from jaxscale_lm.model.attention import CausalSelfAttention
from jaxscale_lm.model.cache import init_cache, update_layer
from jaxscale_lm.model.embeddings import apply_rope, rope_angles
from jaxscale_lm.model.mlp import MLP
from jaxscale_lm.model.transformer import build_model

pytestmark = pytest.mark.unit

CFG = ModelConfig(
    vocab_size=97,
    max_sequence_length=64,
    num_layers=2,
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=4,
    num_key_value_heads=4,
)


def _tokens(batch: int, s: int, seed: int = 0) -> jax.Array:
    return jax.random.randint(jax.random.key(seed), (batch, s), 0, CFG.vocab_size)


class TestShapes:
    def test_mlp_shape(self):
        mlp = MLP(CFG, nnx.Rngs(0))
        out = mlp(jnp.ones((2, 5, CFG.hidden_size)))
        assert out.shape == (2, 5, CFG.hidden_size)

    def test_attention_shape(self):
        attn = CausalSelfAttention(CFG, nnx.Rngs(0))
        out, cache = attn(jnp.ones((2, 5, CFG.hidden_size)))
        assert out.shape == (2, 5, CFG.hidden_size)
        assert cache is None

    def test_transformer_logits_shape_and_dtype(self):
        model = build_model(CFG, seed=0)
        logits, cache = model(_tokens(2, 8))
        assert logits.shape == (2, 8, CFG.vocab_size)
        assert logits.dtype == jnp.float32
        assert cache is None

    def test_bf16_compute_still_f32_logits(self):
        cfg = CFG.model_copy(update={"compute_dtype": "bfloat16"})
        model = build_model(cfg, seed=0)
        logits, _ = model(_tokens(2, 8))
        assert logits.dtype == jnp.float32
        assert bool(jnp.isfinite(logits).all())


class TestCausality:
    def test_future_tokens_do_not_affect_past_logits(self):
        model = build_model(CFG, seed=0)
        tokens = _tokens(1, 12)
        logits_a, _ = model(tokens)
        # Perturb the last 4 tokens; logits at positions < 8 must be identical.
        perturbed = tokens.at[:, 8:].set((tokens[:, 8:] + 1) % CFG.vocab_size)
        logits_b, _ = model(perturbed)
        np.testing.assert_array_equal(np.asarray(logits_a[:, :8]), np.asarray(logits_b[:, :8]))
        # Sanity: the perturbation does change later logits.
        assert not np.allclose(np.asarray(logits_a[:, 8:]), np.asarray(logits_b[:, 8:]))

    def test_prefix_invariance_per_position(self):
        """Logits at position t computed from a length-t prefix must equal
        the same position computed from the full sequence."""
        model = build_model(CFG, seed=0)
        tokens = _tokens(1, 10)
        full_logits, _ = model(tokens)
        for t in (1, 4, 9):
            prefix_logits, _ = model(tokens[:, : t + 1])
            np.testing.assert_allclose(
                np.asarray(prefix_logits[:, t]),
                np.asarray(full_logits[:, t]),
                rtol=1e-5,
                atol=1e-5,
            )


class TestRope:
    def test_rotation_preserves_norm(self):
        x = jax.random.normal(jax.random.key(0), (2, 6, 4, 8))
        pos = jnp.broadcast_to(jnp.arange(6, dtype=jnp.int32)[None, :], (2, 6))
        cos, sin = rope_angles(pos, 8, 10_000.0)
        rotated = apply_rope(x, cos, sin)
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(x), axis=-1),
            np.linalg.norm(np.asarray(rotated), axis=-1),
            rtol=1e-5,
        )

    def test_position_zero_is_identity(self):
        x = jax.random.normal(jax.random.key(0), (1, 1, 2, 8))
        cos, sin = rope_angles(jnp.zeros((1, 1), jnp.int32), 8, 10_000.0)
        np.testing.assert_allclose(np.asarray(apply_rope(x, cos, sin)), np.asarray(x), rtol=1e-6)

    def test_odd_head_dim_rejected(self):
        with pytest.raises(ValueError, match="even"):
            rope_angles(jnp.zeros((1, 1), jnp.int32), 7, 10_000.0)


class TestKVCache:
    def test_init_shapes(self):
        cache = init_cache(CFG, batch_size=3, capacity=16)
        assert len(cache.layers) == CFG.num_layers
        assert cache.layers[0].k.shape == (3, 16, CFG.kv_heads, CFG.head_dim)
        assert int(cache.length) == 0

    def test_capacity_validation(self):
        with pytest.raises(ValueError, match="max_sequence_length"):
            init_cache(CFG, batch_size=1, capacity=CFG.max_sequence_length + 1)
        with pytest.raises(ValueError, match="positive"):
            init_cache(CFG, batch_size=1, capacity=0)

    def test_update_layer_writes_at_offset(self):
        cache = init_cache(CFG, batch_size=1, capacity=8)
        layer = cache.layers[0]
        k_new = jnp.ones((1, 2, CFG.kv_heads, CFG.head_dim))
        v_new = 2 * jnp.ones((1, 2, CFG.kv_heads, CFG.head_dim))
        updated = update_layer(layer, k_new, v_new, jnp.asarray(3, jnp.int32))
        np.testing.assert_array_equal(np.asarray(updated.k[:, 3:5]), np.asarray(k_new))
        np.testing.assert_array_equal(np.asarray(updated.v[:, 3:5]), np.asarray(v_new))
        assert np.asarray(updated.k[:, :3]).sum() == 0
        assert np.asarray(updated.k[:, 5:]).sum() == 0

    def test_cache_length_advances(self):
        model = build_model(CFG, seed=0)
        cache = init_cache(CFG, batch_size=1, capacity=16)
        tokens = _tokens(1, 5)
        _, cache = model(tokens, cache=cache)
        assert cache is not None and int(cache.length) == 5
        _, cache = model(_tokens(1, 1, seed=1), cache=cache)
        assert cache is not None and int(cache.length) == 6

    def test_cached_decode_matches_full_forward(self):
        """Prefill + token-by-token decode must reproduce full-sequence logits."""
        model = build_model(CFG, seed=0)
        tokens = _tokens(2, 10)
        full_logits, _ = model(tokens)

        cache = init_cache(CFG, batch_size=2, capacity=16)
        prefill_logits, cache = model(tokens[:, :6], cache=cache)
        np.testing.assert_allclose(
            np.asarray(prefill_logits), np.asarray(full_logits[:, :6]), rtol=2e-5, atol=2e-5
        )
        for t in range(6, 10):
            step_logits, cache = model(tokens[:, t : t + 1], cache=cache)
            np.testing.assert_allclose(
                np.asarray(step_logits[:, 0]),
                np.asarray(full_logits[:, t]),
                rtol=2e-5,
                atol=2e-5,
            )

    def test_batch_dimension_preserved(self):
        model = build_model(CFG, seed=0)
        cache = init_cache(CFG, batch_size=4, capacity=8)
        logits, cache = model(_tokens(4, 3), cache=cache)
        assert logits.shape[0] == 4
        assert cache is not None and cache.layers[0].k.shape[0] == 4

    def test_mismatched_cache_args_rejected(self):
        attn = CausalSelfAttention(CFG, nnx.Rngs(0))
        cache = init_cache(CFG, batch_size=1, capacity=8)
        with pytest.raises(ValueError, match="together"):
            attn(jnp.ones((1, 1, CFG.hidden_size)), cache=cache.layers[0], cache_length=None)


class TestParameters:
    def test_param_count_formula_tied(self):
        model = build_model(CFG, seed=0)
        h, v, i, layers = CFG.hidden_size, CFG.vocab_size, CFG.intermediate_size, CFG.num_layers
        per_layer = 4 * h * h + 2 * h * i + 2 * h  # qkvo + mlp + 2 norms
        expected = v * h + layers * per_layer + h  # embed + blocks + final norm
        assert model.num_params() == expected

    def test_untied_adds_head(self):
        cfg = CFG.model_copy(update={"tie_embeddings": False})
        tied = build_model(CFG, seed=0).num_params()
        untied = build_model(cfg, seed=0).num_params()
        assert untied == tied + CFG.vocab_size * CFG.hidden_size

    def test_preset_parameter_counts_stable(self):
        """Regression guard: preset sizes stay in their documented ranges."""
        from pathlib import Path

        from jaxscale_lm.config import load_config

        config_dir = Path(__file__).parent.parent.parent / "configs"
        tiny = build_model(load_config(config_dir / "model" / "tiny.yaml").model, seed=0)
        assert 1_000_000 <= tiny.num_params() <= 5_000_000

    def test_dropout_changes_only_in_training_mode(self):
        cfg = CFG.model_copy(update={"dropout_rate": 0.5})
        model = build_model(cfg, seed=0)
        tokens = _tokens(1, 8)
        det_a, _ = model(tokens, deterministic=True)
        det_b, _ = model(tokens, deterministic=True)
        np.testing.assert_array_equal(np.asarray(det_a), np.asarray(det_b))
        stoch, _ = model(tokens, deterministic=False, rngs=nnx.Rngs(dropout=1))
        assert not np.allclose(np.asarray(det_a), np.asarray(stoch))

    def test_nondeterministic_dropout_requires_rngs(self):
        cfg = CFG.model_copy(update={"dropout_rate": 0.5})
        model = build_model(cfg, seed=0)
        with pytest.raises(ValueError):
            model(_tokens(1, 4), deterministic=False)
