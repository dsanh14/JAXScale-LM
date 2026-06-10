"""The inference engine: checkpoint loading, validation, generation, timing.

One engine instance owns one loaded model (params + graphdef + tokenizer)
and the jitted prefill/decode functions. Jit caches are keyed by input
shapes, so a new (batch, prompt_length) pair triggers one compilation and is
fast afterwards; ``warmup()`` pre-compiles a representative shape so serving
readiness implies a compiled path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from jaxscale_lm.config import Config, InferenceConfig
from jaxscale_lm.data.tokenizer import Tokenizer, build_tokenizer
from jaxscale_lm.inference.decode import make_cached_decode_fn, make_naive_decode_fn
from jaxscale_lm.inference.generate import (
    GenerationTiming,
    cached_generate,
    naive_generate,
)
from jaxscale_lm.inference.prefill import make_prefill_fn
from jaxscale_lm.inference.sampling import SamplingParams
from jaxscale_lm.model.cache import init_cache
from jaxscale_lm.model.transformer import build_model
from jaxscale_lm.training.checkpoint import Checkpointer, read_metadata, resolve_checkpoint
from jaxscale_lm.training.optimizer import build_optimizer
from jaxscale_lm.training.state import create_train_state
from jaxscale_lm.utils.logging import get_logger, log_event
from jaxscale_lm.utils.seed import make_key

_logger = get_logger("engine")


@dataclass(frozen=True)
class GenerationResult:
    """Everything the CLI and serving layer report about one generation."""

    generated_text: str
    generated_token_ids: list[int]
    prompt_tokens: int
    generated_tokens: int
    timing: GenerationTiming
    cache_enabled: bool
    checkpoint_step: int

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / max(self.timing.decode_s, 1e-9)

    @property
    def time_to_first_token_s(self) -> float:
        return self.timing.prefill_s


class InferenceEngine:
    """Loads a checkpoint and serves generation requests."""

    def __init__(
        self,
        config: Config,
        params: nnx.State,
        graphdef: nnx.GraphDef,
        tokenizer: Tokenizer,
        checkpoint_step: int,
    ) -> None:
        self.config = config
        self.model_config = config.model
        self.tokenizer = tokenizer
        self.checkpoint_step = checkpoint_step
        self._params = params
        self._graphdef = graphdef
        self._prefill = jax.jit(make_prefill_fn(graphdef))
        self._decode = jax.jit(make_cached_decode_fn(graphdef))
        self._naive = jax.jit(make_naive_decode_fn(graphdef))

    # -- construction ---------------------------------------------------
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> InferenceEngine:
        """Build an engine from a checkpoint directory (uses its saved config)."""
        ref = resolve_checkpoint(checkpoint_path)
        step, metadata = read_metadata(ref.root, ref.step)
        config = Config.model_validate(metadata["config"])
        config = config.model_copy(
            update={
                "checkpoint": config.checkpoint.model_copy(update={"directory": ref.root})
            }
        )

        model = build_model(config.model, config.project.seed)
        tx, _ = build_optimizer(config.optimizer, config.training.max_steps)
        _, template = create_train_state(model, tx, make_key(config.project.seed))
        checkpointer = Checkpointer(ref.root, config)
        try:
            state, _ = checkpointer.restore(template, step)
        finally:
            checkpointer.close()
        graphdef, _ = nnx.split(model)
        tokenizer = build_tokenizer(config.tokenizer)
        log_event(
            _logger,
            "engine loaded",
            checkpoint=str(ref.root),
            step=step,
            parameters=model.num_params(),
            platform=jax.default_backend(),
        )
        return cls(config, state.params, graphdef, tokenizer, step)

    # -- validation -------------------------------------------------------
    def _validate(self, prompt_tokens: int, options: InferenceConfig) -> None:
        """Validate a request against the model context and sampling rules."""
        max_len = self.model_config.max_sequence_length
        if prompt_tokens == 0:
            raise ValueError("Prompt encoded to zero tokens; provide a non-empty prompt.")
        if prompt_tokens >= max_len:
            raise ValueError(
                f"Prompt is {prompt_tokens} tokens but the model context is {max_len}; "
                f"shorten the prompt."
            )
        needed = prompt_tokens + options.max_new_tokens
        if needed > max_len:
            raise ValueError(
                f"prompt ({prompt_tokens}) + max_new_tokens ({options.max_new_tokens}) "
                f"= {needed} exceeds the model context ({max_len}); reduce one of them."
            )
        sampling = SamplingParams(
            do_sample=options.do_sample,
            temperature=options.temperature,
            top_k=options.top_k,
            top_p=options.top_p,
            repetition_penalty=options.repetition_penalty,
        )
        sampling.validate(self.model_config.vocab_size)

    # -- generation ---------------------------------------------------------
    def generate(self, prompt: str, options: InferenceConfig) -> GenerationResult:
        """Generate a completion for one prompt."""
        prompt_ids_list = self.tokenizer.encode(prompt)
        self._validate(len(prompt_ids_list), options)
        # Capacity is pinned to the model context so the decode step (whose
        # jit cache is keyed by the cache shape) compiles exactly once for
        # every request; per-request capacities would recompile constantly.
        capacity = self.model_config.max_sequence_length
        prompt_ids = jnp.asarray([prompt_ids_list], dtype=jnp.int32)
        sampling = SamplingParams(
            do_sample=options.do_sample,
            temperature=options.temperature,
            top_k=options.top_k,
            top_p=options.top_p,
            repetition_penalty=options.repetition_penalty,
        )
        key = make_key(options.seed)

        if options.use_kv_cache:
            cache = init_cache(self.model_config, batch_size=1, capacity=capacity)
            output = cached_generate(
                self._prefill,
                self._decode,
                self._params,
                prompt_ids,
                cache,
                max_new_tokens=options.max_new_tokens,
                sampling=sampling,
                key=key,
                eos_id=self.tokenizer.eos_id,
                pad_id=self.tokenizer.pad_id,
                vocab_size=self.model_config.vocab_size,
            )
        else:
            output = naive_generate(
                self._naive,
                self._params,
                prompt_ids,
                capacity=capacity,
                max_new_tokens=options.max_new_tokens,
                sampling=sampling,
                key=key,
                eos_id=self.tokenizer.eos_id,
                pad_id=self.tokenizer.pad_id,
                vocab_size=self.model_config.vocab_size,
            )

        row = output.token_ids[0].tolist()
        # Trim at EOS (everything after is pad by construction).
        if self.tokenizer.eos_id in row:
            row = row[: row.index(self.tokenizer.eos_id)]
        return GenerationResult(
            generated_text=self.tokenizer.decode(row),
            generated_token_ids=row,
            prompt_tokens=len(prompt_ids_list),
            generated_tokens=len(row),
            timing=output.timing,
            cache_enabled=options.use_kv_cache,
            checkpoint_step=self.checkpoint_step,
        )

    def warmup(self, max_new_tokens: int = 4) -> float:
        """Pre-compile the cached decode path (and one prefill shape).

        The decode step has a single shape for all requests (capacity is
        pinned), so after warmup every request hits a compiled decode; only
        unseen *prompt lengths* still pay a prefill compilation. Returns wall
        seconds spent (dominated by XLA compilation).
        """
        import time

        options = InferenceConfig(max_new_tokens=max_new_tokens, do_sample=False)
        start = time.perf_counter()
        self.generate("warmup", options)
        elapsed = time.perf_counter() - start
        log_event(_logger, "warmup complete", seconds=round(elapsed, 3))
        return elapsed
