"""Configuration validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jaxscale_lm.config import Config, ModelConfig, load_config, save_resolved_config

pytestmark = pytest.mark.unit

CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"


class TestModelConfig:
    def test_defaults_valid(self):
        cfg = ModelConfig()
        assert cfg.head_dim == cfg.hidden_size // cfg.num_attention_heads

    def test_hidden_size_not_divisible_by_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            ModelConfig(hidden_size=100, num_attention_heads=3)

    def test_heads_not_divisible_by_kv_heads(self):
        with pytest.raises(ValueError, match="num_key_value_heads"):
            ModelConfig(num_attention_heads=4, num_key_value_heads=3)

    def test_kv_heads_default_to_heads(self):
        cfg = ModelConfig(num_attention_heads=8, hidden_size=128)
        assert cfg.kv_heads == 8

    def test_odd_head_dim_rejected(self):
        with pytest.raises(ValueError, match="even"):
            ModelConfig(hidden_size=12, num_attention_heads=4)  # head_dim 3

    def test_negative_dropout_rejected(self):
        with pytest.raises(ValueError):
            ModelConfig(dropout_rate=-0.1)

    def test_unknown_dtype_rejected(self):
        with pytest.raises(ValueError):
            ModelConfig(compute_dtype="float64")  # type: ignore[arg-type]


class TestCrossValidation:
    def test_sequence_length_exceeds_context(self):
        with pytest.raises(ValueError, match="max_sequence_length"):
            Config.model_validate(
                {
                    "data": {"sequence_length": 512},
                    "model": {"max_sequence_length": 256},
                }
            )

    def test_vocab_mismatch_rejected(self):
        with pytest.raises(ValueError, match="vocab_size"):
            Config.model_validate(
                {"tokenizer": {"kind": "bpe", "vocab_size": 1000}, "model": {"vocab_size": 999}}
            )

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError):
            Config.model_validate({"model": {"hiden_size": 64}})

    def test_zero_accumulation_rejected(self):
        with pytest.raises(ValueError):
            Config.model_validate({"training": {"gradient_accumulation_steps": 0}})

    def test_zero_checkpoint_interval_rejected(self):
        with pytest.raises(ValueError):
            Config.model_validate({"checkpoint": {"interval_steps": 0}})

    def test_top_p_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            Config.model_validate({"inference": {"top_p": 1.5}})

    def test_negative_top_k_rejected(self):
        with pytest.raises(ValueError):
            Config.model_validate({"inference": {"top_k": -1}})


class TestYamlLoading:
    @pytest.mark.parametrize(
        "rel",
        [
            "model/tiny.yaml",
            "model/small.yaml",
            "train/cpu_smoke.yaml",
            "train/single_device.yaml",
            "train/multi_device.yaml",
            "inference/default.yaml",
            "benchmark/default.yaml",
        ],
    )
    def test_shipped_configs_load(self, rel: str):
        cfg = load_config(CONFIG_DIR / rel)
        assert isinstance(cfg, Config)

    def test_defaults_merge_order(self, tmp_path: Path):
        base = tmp_path / "base.yaml"
        base.write_text("model: {hidden_size: 64, num_attention_heads: 4}\n")
        child = tmp_path / "child.yaml"
        child.write_text(f"defaults: [{base}]\nmodel: {{hidden_size: 128}}\n")
        cfg = load_config(child)
        assert cfg.model.hidden_size == 128  # child wins
        assert cfg.model.num_attention_heads == 4  # base preserved

    def test_circular_defaults_detected(self, tmp_path: Path):
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text(f"defaults: [{b}]\n")
        b.write_text(f"defaults: [{a}]\n")
        with pytest.raises(ValueError, match=r"[Cc]ircular"):
            load_config(a)

    def test_missing_file_actionable_error(self):
        with pytest.raises(FileNotFoundError, match=r"no_such_config\.yaml"):
            load_config("no_such_config.yaml")

    def test_resolved_config_round_trips(self, tmp_path: Path):
        cfg = load_config(CONFIG_DIR / "train" / "cpu_smoke.yaml")
        out = tmp_path / "resolved.yaml"
        save_resolved_config(cfg, out)
        reloaded = Config.model_validate(yaml.safe_load(out.read_text()))
        assert reloaded == cfg
