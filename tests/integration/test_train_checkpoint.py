"""Integration tests: smoke training, exact checkpoint resumption, evaluation.

These use the real cpu_smoke config (redirected to a temp artifacts dir) so
they exercise the same path as `make train-smoke`.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from jaxscale_lm.config import Config
from jaxscale_lm.training.trainer import Trainer

pytestmark = pytest.mark.integration


def _with_updates(config: Config, **training_updates) -> Config:
    return config.model_copy(
        update={"training": config.training.model_copy(update=training_updates)}
    )


class TestSmokeTraining:
    def test_full_smoke_run(self, tmp_artifacts: Config):
        trainer = Trainer(tmp_artifacts)
        summary = trainer.train()
        assert np.isfinite(summary["loss"])
        assert summary["perplexity"] > 0
        assert int(trainer.state.step) == tmp_artifacts.training.max_steps
        # checkpoints exist
        assert (tmp_artifacts.checkpoint_dir / "resolved_config.yaml").exists()


class TestExactResumption:
    def test_interrupted_equals_uninterrupted(self, tmp_path, smoke_config: Config):
        """Train N+M straight vs train N, save, restore, train M: identical."""
        n, m = 5, 5

        def make_config(name: str) -> Config:
            return smoke_config.model_copy(
                update={
                    "project": smoke_config.project.model_copy(
                        update={"artifacts_dir": tmp_path / name, "run_name": name}
                    )
                }
            )

        # Uninterrupted N+M run.
        cfg_full = _with_updates(make_config("full"), max_steps=n + m)
        trainer_full = Trainer(cfg_full)
        trainer_full.train()

        # Interrupted: same N+M config stopped at N (an interrupted run keeps
        # its schedule horizon — a shorter max_steps would change the LR
        # trajectory), checkpoint, fresh trainer, M more steps.
        cfg_n = _with_updates(make_config("resumed"), max_steps=n + m)
        trainer_n = Trainer(cfg_n)
        trainer_n.train(until_step=n)  # saves at step 5 (checkpoint.interval_steps=5)
        del trainer_n

        cfg_resume = _with_updates(make_config("resumed"), max_steps=n + m)
        trainer_resumed = Trainer(cfg_resume)
        restored_step = trainer_resumed.resume()
        assert restored_step == n
        trainer_resumed.train()

        # Compare: step, parameters, optimizer state, next-step eval loss.
        assert int(trainer_resumed.state.step) == int(trainer_full.state.step) == n + m
        for a, b in zip(
            jax.tree.leaves(trainer_full.state.params),
            jax.tree.leaves(trainer_resumed.state.params),
            strict=True,
        ):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-7)
        for a, b in zip(
            jax.tree.leaves(trainer_full.state.opt_state),
            jax.tree.leaves(trainer_resumed.state.opt_state),
            strict=True,
        ):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-7)

        eval_full = trainer_full.evaluate()
        eval_resumed = trainer_resumed.evaluate()
        assert eval_full["loss"] == pytest.approx(eval_resumed["loss"], rel=1e-6)

    def test_restore_specific_step(self, tmp_artifacts: Config):
        trainer = Trainer(tmp_artifacts)
        trainer.train()  # checkpoints at steps 5 and 10
        steps = trainer.checkpointer.all_steps()
        assert 5 in steps and 10 in steps

        fresh = Trainer(tmp_artifacts)
        restored = fresh.resume(step=5)
        assert restored == 5
        fresh.checkpointer.close()

    def test_restore_missing_checkpoint_actionable(self, tmp_artifacts: Config):
        trainer = Trainer(tmp_artifacts)
        try:
            with pytest.raises(FileNotFoundError, match="No checkpoint"):
                trainer.resume()
        finally:
            trainer.checkpointer.close()

    def test_incompatible_config_rejected(self, tmp_artifacts: Config):
        trainer = Trainer(tmp_artifacts)
        trainer.train()

        bad = tmp_artifacts.model_copy(
            update={
                "model": tmp_artifacts.model.model_copy(update={"hidden_size": 128}),
                "tokenizer": tmp_artifacts.tokenizer,
            }
        )
        other = Trainer(bad)
        try:
            with pytest.raises(ValueError, match="incompatible"):
                other.resume()
        finally:
            other.checkpointer.close()
