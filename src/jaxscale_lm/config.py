"""Typed, validated configuration for all JAXScale-LM workflows.

Configuration is authored in YAML, optionally composed from other YAML files
via a top-level ``defaults`` list (merged shallowly-by-section, later files
win), and validated into immutable Pydantic models *before* any model or
device initialization happens.

The fully resolved configuration is saved alongside every training run so
that experiments are reproducible from artifacts alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DTypeName = Literal["float32", "bfloat16", "float16"]


class _StrictModel(BaseModel):
    """Base model: immutable and intolerant of unknown keys (catches typos)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ProjectConfig(_StrictModel):
    name: str = "jaxscale-lm"
    run_name: str = "dev"
    artifacts_dir: Path = Path("artifacts")
    seed: int = 0


class DataConfig(_StrictModel):
    source: Literal["synthetic", "local_text", "tinystories"] = "synthetic"
    local_path: Path | None = None
    cache_dir: Path = Path("artifacts/data")
    sequence_length: int = Field(default=64, gt=0)
    batch_size: int = Field(
        default=4, gt=0, description="Per-step microbatch size (global across data axis)"
    )
    validation_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    shuffle: bool = True
    drop_remainder: bool = True
    max_train_documents: int | None = Field(
        default=None, gt=0, description="Optional cap on documents loaded, for fast experiments"
    )
    synthetic_num_documents: int = Field(default=64, gt=0)
    synthetic_document_length: int = Field(default=256, gt=0)

    @model_validator(mode="after")
    def _check_local_path(self) -> Self:
        if self.source == "local_text" and self.local_path is None:
            raise ValueError(
                "data.source is 'local_text' but data.local_path is not set. "
                "Point it at a directory of .txt files or a single .txt file."
            )
        return self


class TokenizerConfig(_StrictModel):
    kind: Literal["byte", "bpe"] = "byte"
    vocab_size: int = Field(default=259, gt=0, description="Total vocab incl. special tokens")
    path: Path | None = Field(default=None, description="Trained tokenizer file (bpe only)")

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.kind == "byte" and self.vocab_size != 259:
            raise ValueError(
                f"Byte tokenizer has a fixed vocab of 259 (256 bytes + PAD/BOS/EOS); "
                f"got tokenizer.vocab_size={self.vocab_size}."
            )
        return self


class ModelConfig(_StrictModel):
    vocab_size: int = Field(default=259, gt=0)
    max_sequence_length: int = Field(default=256, gt=0)
    num_layers: int = Field(default=2, gt=0)
    hidden_size: int = Field(default=128, gt=0)
    intermediate_size: int = Field(default=512, gt=0)
    num_attention_heads: int = Field(default=4, gt=0)
    num_key_value_heads: int | None = Field(
        default=None, description="Defaults to num_attention_heads (no grouped-query attention)"
    )
    dropout_rate: float = Field(default=0.0, ge=0.0, lt=1.0)
    attention_dropout_rate: float = Field(default=0.0, ge=0.0, lt=1.0)
    normalization_epsilon: float = Field(default=1e-6, gt=0.0)
    parameter_dtype: DTypeName = "float32"
    compute_dtype: DTypeName = "float32"
    tie_embeddings: bool = True
    use_bias: bool = False
    initializer_range: float = Field(default=0.02, gt=0.0)
    rope_theta: float = Field(default=10_000.0, gt=0.0)

    @property
    def kv_heads(self) -> int:
        return self.num_key_value_heads or self.num_attention_heads

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"model.hidden_size ({self.hidden_size}) must be divisible by "
                f"model.num_attention_heads ({self.num_attention_heads})."
            )
        kv = self.kv_heads
        if self.num_attention_heads % kv != 0:
            raise ValueError(
                f"model.num_attention_heads ({self.num_attention_heads}) must be divisible "
                f"by model.num_key_value_heads ({kv})."
            )
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"Head dimension ({self.head_dim}) must be even for rotary embeddings; "
                f"adjust hidden_size or num_attention_heads."
            )
        return self


class OptimizerConfig(_StrictModel):
    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    beta2: float = Field(default=0.95, ge=0.0, lt=1.0)
    eps: float = Field(default=1e-8, gt=0.0)
    grad_clip_norm: float | None = Field(default=1.0, gt=0.0)
    warmup_steps: int = Field(default=10, ge=0)
    schedule: Literal["cosine", "linear", "constant"] = "cosine"
    min_learning_rate_ratio: float = Field(default=0.1, ge=0.0, le=1.0)


class TrainingConfig(_StrictModel):
    max_steps: int = Field(default=100, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    log_interval: int = Field(default=10, gt=0)


class DistributedConfig(_StrictModel):
    data_axis_size: int = Field(
        default=-1,
        description="-1 means 'all available devices'; otherwise must divide device count",
    )
    model_axis_size: int = Field(default=1, gt=0)
    axis_names: tuple[str, str] = ("data", "model")


class CheckpointConfig(_StrictModel):
    interval_steps: int = Field(default=50, gt=0)
    max_to_keep: int = Field(default=3, gt=0)
    keep_best: bool = True
    best_metric: str = "validation_loss"
    directory: Path | None = Field(
        default=None, description="Defaults to <artifacts_dir>/checkpoints/<run_name>"
    )


class EvaluationConfig(_StrictModel):
    interval_steps: int = Field(default=50, gt=0)
    num_batches: int = Field(default=8, gt=0)


class InferenceConfig(_StrictModel):
    max_new_tokens: int = Field(default=64, gt=0)
    temperature: float = Field(default=1.0, gt=0.0)
    top_k: int = Field(default=0, ge=0, description="0 disables top-k filtering")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    do_sample: bool = False
    seed: int = 0
    use_kv_cache: bool = True
    repetition_penalty: float | None = Field(default=None, gt=0.0)


class ServingConfig(_StrictModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, gt=0, lt=65536)
    warmup: bool = True
    max_prompt_tokens: int = Field(default=1024, gt=0)
    max_new_tokens_limit: int = Field(default=512, gt=0)
    registry_path: Path | None = Field(
        default=None, description="Defaults to <artifacts_dir>/registry.json"
    )


class BenchmarkConfig(_StrictModel):
    output_dir: Path | None = Field(
        default=None, description="Defaults to <artifacts_dir>/benchmarks"
    )
    warmup_iterations: int = Field(default=3, ge=0)
    measure_iterations: int = Field(default=10, gt=0)
    seed: int = 0
    suites: tuple[str, ...] = ("compilation", "training", "prefill", "decode", "e2e", "cache")
    batch_sizes: tuple[int, ...] = (1, 2, 4)
    prompt_lengths: tuple[int, ...] = (16, 64)
    generate_lengths: tuple[int, ...] = (32,)
    sequence_lengths: tuple[int, ...] = (64, 128)
    precisions: tuple[DTypeName, ...] = ("float32",)

    @model_validator(mode="after")
    def _check(self) -> Self:
        known = {"compilation", "training", "prefill", "decode", "e2e", "cache", "sharding"}
        unknown = set(self.suites) - known
        if unknown:
            raise ValueError(f"Unknown benchmark suites {sorted(unknown)}; known: {sorted(known)}")
        return self


class LoggingConfig(_StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_format: bool = False


class Config(_StrictModel):
    """Root configuration: every workflow reads the sections it needs."""

    project: ProjectConfig = ProjectConfig()
    data: DataConfig = DataConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    training: TrainingConfig = TrainingConfig()
    distributed: DistributedConfig = DistributedConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    inference: InferenceConfig = InferenceConfig()
    serving: ServingConfig = ServingConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
    logging: LoggingConfig = LoggingConfig()

    @model_validator(mode="after")
    def _cross_checks(self) -> Self:
        if self.data.sequence_length > self.model.max_sequence_length:
            raise ValueError(
                f"data.sequence_length ({self.data.sequence_length}) exceeds "
                f"model.max_sequence_length ({self.model.max_sequence_length})."
            )
        if self.tokenizer.vocab_size != self.model.vocab_size:
            raise ValueError(
                f"tokenizer.vocab_size ({self.tokenizer.vocab_size}) must equal "
                f"model.vocab_size ({self.model.vocab_size})."
            )
        return self

    # -- derived paths ------------------------------------------------------
    @property
    def checkpoint_dir(self) -> Path:
        if self.checkpoint.directory is not None:
            return self.checkpoint.directory
        return self.project.artifacts_dir / "checkpoints" / self.project.run_name

    @property
    def benchmark_dir(self) -> Path:
        return self.benchmark.output_dir or (self.project.artifacts_dir / "benchmarks")

    @property
    def registry_path(self) -> Path:
        return self.serving.registry_path or (self.project.artifacts_dir / "registry.json")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_defaults(path: Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Circular config 'defaults' chain involving {path}")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Top level of {path} must be a YAML mapping, got {type(raw).__name__}")

    defaults = raw.pop("defaults", [])
    merged: dict[str, Any] = {}
    for default in defaults:
        default_path = Path(default)
        if not default_path.is_absolute():
            default_path = path.parent / default_path
        merged = _deep_merge(merged, _load_yaml_with_defaults(default_path, _seen | {path}))
    return _deep_merge(merged, raw)


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file (with ``defaults`` composition)."""
    return Config.model_validate(_load_yaml_with_defaults(Path(path)))


def save_resolved_config(config: Config, path: Path) -> None:
    """Write the fully resolved configuration as YAML for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
