"""Serving-layer unit tests: registry transitions, schema validation, metrics."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jaxscale_lm.serving.registry import ModelEntry, ModelRegistry, ModelStatus
from jaxscale_lm.serving.schemas import GenerateRequest

pytestmark = pytest.mark.unit


def _entry(model_id: str = "m1") -> ModelEntry:
    return ModelEntry(
        model_id=model_id,
        version=1,
        checkpoint_path="/tmp/ckpt",
        training_step=10,
        parameter_count=1000,
        max_sequence_length=128,
        precision="float32",
    )


class TestRegistry:
    def test_register_and_get(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(_entry())
        assert reg.get("m1").status == ModelStatus.REGISTERED.value

    def test_duplicate_rejected(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(_entry())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_entry())

    def test_unknown_id_actionable(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry.json")
        with pytest.raises(KeyError, match="Unknown model id"):
            reg.get("nope")

    def test_valid_lifecycle(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(_entry())
        reg.set_status("m1", ModelStatus.LOADING)
        reg.set_status("m1", ModelStatus.READY)
        reg.set_status("m1", ModelStatus.UNLOADED)
        reg.set_status("m1", ModelStatus.LOADING)
        reg.set_status("m1", ModelStatus.FAILED)
        assert reg.get("m1").status == "FAILED"

    def test_illegal_transition_rejected(self, tmp_path):
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register(_entry())
        with pytest.raises(ValueError, match="Illegal status transition"):
            reg.set_status("m1", ModelStatus.READY)  # REGISTERED -> READY skips LOADING

    def test_persistence_round_trip(self, tmp_path):
        path = tmp_path / "registry.json"
        reg = ModelRegistry(path)
        reg.register(_entry())
        reg.set_status("m1", ModelStatus.LOADING)

        reloaded = ModelRegistry(path)
        assert reloaded.get("m1").status == "LOADING"
        # file is valid JSON with a version marker
        raw = json.loads(path.read_text())
        assert raw["registry_version"] == 1

    def test_unsupported_version_rejected(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(json.dumps({"registry_version": 99, "models": {}}))
        with pytest.raises(ValueError, match="registry version"):
            ModelRegistry(path)


class TestSchemas:
    def test_valid_request(self):
        req = GenerateRequest(prompt="hello", max_new_tokens=10)
        assert req.use_kv_cache is True

    def test_empty_prompt_rejected(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="")

    def test_bad_temperature_rejected(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="x", temperature=0.0)

    def test_bad_top_p_rejected(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="x", top_p=2.0)

    def test_negative_max_new_tokens_rejected(self):
        with pytest.raises(ValidationError):
            GenerateRequest(prompt="x", max_new_tokens=-5)
