"""End-to-end serving tests: train a tiny model, serve it via ASGI, exercise
every endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from jaxscale_lm.config import Config
from jaxscale_lm.serving.app import create_app
from jaxscale_lm.training.trainer import Trainer

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory) -> tuple[Config, str]:
    """Train the smoke model once for all serving tests."""
    import os

    from jaxscale_lm.config import load_config

    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
    base = load_config(os.path.join(config_path, "train", "cpu_smoke.yaml"))
    artifacts = tmp_path_factory.mktemp("serving_artifacts")
    config = base.model_copy(
        update={"project": base.project.model_copy(update={"artifacts_dir": artifacts})}
    )
    trainer = Trainer(config)
    trainer.train()
    return config, str(config.checkpoint_dir / "latest")


@pytest.fixture()
def client(trained_checkpoint) -> TestClient:
    config, checkpoint = trained_checkpoint
    app = create_app(config, initial_checkpoint=checkpoint)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


class TestLifecycleEndpoints:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ready_after_startup_warmup(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_models_listed_ready(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200
        models = response.json()["models"]
        assert len(models) == 1
        assert models[0]["status"] == "READY"
        assert models[0]["parameter_count"] > 0

    def test_model_detail_and_404(self, client):
        model_id = client.get("/v1/models").json()["models"][0]["model_id"]
        assert client.get(f"/v1/models/{model_id}").status_code == 200
        assert client.get("/v1/models/does-not-exist").status_code == 404

    def test_unload_then_unready(self, trained_checkpoint):
        config, checkpoint = trained_checkpoint
        app = create_app(config, initial_checkpoint=checkpoint)
        with TestClient(app) as c:
            model_id = c.get("/v1/models").json()["models"][0]["model_id"]
            assert c.post("/v1/models/unload", json={"model_id": model_id}).status_code == 200
            assert c.get("/ready").status_code == 503
            assert (
                c.post("/v1/generate", json={"prompt": "hi", "max_new_tokens": 4}).status_code
                == 503
            )

    def test_load_bad_path_404(self, client):
        response = client.post("/v1/models/load", json={"checkpoint_path": "/nonexistent/ckpt"})
        assert response.status_code == 404


class TestGeneration:
    def test_generate_returns_text_and_timings(self, client):
        response = client.post(
            "/v1/generate",
            json={"prompt": "the cat sat", "max_new_tokens": 8, "use_kv_cache": True},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert isinstance(body["generated_text"], str)
        assert body["generated_tokens"] == len(body["generated_token_ids"])
        assert body["prompt_tokens"] > 0
        assert body["total_latency_ms"] > 0
        assert body["tokens_per_second"] > 0
        assert body["cache_enabled"] is True
        assert body["device_platform"] == "cpu"
        assert body["request_id"]
        assert response.headers["x-request-id"] == body["request_id"]

    def test_generate_no_cache_path(self, client):
        response = client.post(
            "/v1/generate",
            json={"prompt": "the cat", "max_new_tokens": 4, "use_kv_cache": False},
        )
        assert response.status_code == 200
        assert response.json()["cache_enabled"] is False

    def test_greedy_deterministic_via_api(self, client):
        payload = {"prompt": "once upon", "max_new_tokens": 8, "do_sample": False}
        a = client.post("/v1/generate", json=payload).json()
        b = client.post("/v1/generate", json=payload).json()
        assert a["generated_token_ids"] == b["generated_token_ids"]

    def test_overlong_generation_rejected(self, client, trained_checkpoint):
        config, _ = trained_checkpoint
        response = client.post(
            "/v1/generate",
            json={
                "prompt": "x",
                "max_new_tokens": config.serving.max_new_tokens_limit + 1,
            },
        )
        assert response.status_code == 400
        assert "limit" in response.json()["detail"]

    def test_context_overflow_rejected(self, client, trained_checkpoint):
        config, _ = trained_checkpoint
        # max_new_tokens within server limit but prompt+gen exceeds context.
        max_len = config.model.max_sequence_length
        response = client.post(
            "/v1/generate",
            json={"prompt": "y" * (max_len - 2), "max_new_tokens": 10},
        )
        assert response.status_code == 400

    def test_invalid_body_422(self, client):
        assert client.post("/v1/generate", json={"prompt": ""}).status_code == 422
        assert (
            client.post("/v1/generate", json={"prompt": "x", "temperature": -1.0}).status_code
            == 422
        )


class TestMetricsEndpoint:
    def test_metrics_exposed_and_updated(self, client):
        client.post("/v1/generate", json={"prompt": "hello", "max_new_tokens": 4})
        response = client.get("/metrics")
        assert response.status_code == 200
        text = response.text
        assert "jaxscale_requests_total" in text
        assert "jaxscale_generated_tokens_total" in text
        assert "jaxscale_request_latency_seconds" in text
