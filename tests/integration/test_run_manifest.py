"""Integration tests: per-run reproducibility manifest.

A training run must leave an audit trail sufficient to reproduce it:
resolved config, environment, git state, and the metric history — per
`.claude/skills/reproducibility.md`.
"""

from __future__ import annotations

import json

import pytest
import yaml

from jaxscale_lm.config import Config
from jaxscale_lm.training.trainer import Trainer

pytestmark = pytest.mark.integration


class TestRunManifest:
    def test_training_writes_complete_manifest(self, tmp_artifacts: Config):
        trainer = Trainer(tmp_artifacts)
        trainer.train()

        run_dir = trainer.manifest.run_dir
        assert run_dir.parent == tmp_artifacts.project.artifacts_dir / "runs"

        # Resolved config round-trips to the executed configuration.
        resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
        assert resolved["training"]["max_steps"] == tmp_artifacts.training.max_steps
        assert resolved["project"]["seed"] == tmp_artifacts.project.seed

        environment = json.loads((run_dir / "environment.json").read_text())
        for key in ("python_version", "jax_version", "platform", "device_count"):
            assert key in environment

        git = json.loads((run_dir / "git.json").read_text())
        assert set(git) >= {"git_commit", "git_dirty", "git_branch"}

        run = json.loads((run_dir / "run.json").read_text())
        assert run["run_id"] == trainer.manifest.run_id
        assert run["seed"] == tmp_artifacts.project.seed
        assert run["checkpoint_dir"] == str(tmp_artifacts.checkpoint_dir)

        # The checkpoint linkage resolves to the real checkpoint directory.
        link = run_dir / "checkpoints"
        if link.is_symlink():  # skipped only on symlink-less filesystems
            assert (link / "resolved_config.yaml").exists()

    def test_metrics_jsonl_records_training_history(self, tmp_artifacts: Config):
        trainer = Trainer(tmp_artifacts)
        summary = trainer.train()

        lines = (trainer.manifest.run_dir / "metrics.jsonl").read_text().splitlines()
        events = [json.loads(line) for line in lines]
        by_event = {e["event"] for e in events}
        assert {"trainer_initialized", "train_step", "final_evaluation"} <= by_event

        steps = [e for e in events if e["event"] == "train_step"]
        assert steps and all("loss" in e and "learning_rate" in e for e in steps)
        # Steps are recorded in execution order.
        assert [e["step"] for e in steps] == sorted(e["step"] for e in steps)

        final = [e for e in events if e["event"] == "final_evaluation"][-1]
        assert final["loss"] == pytest.approx(summary["loss"])
