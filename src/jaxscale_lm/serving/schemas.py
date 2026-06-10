"""Pydantic request/response schemas for the serving API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=64, gt=0)
    temperature: float = Field(default=1.0, gt=0.0)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    do_sample: bool = False
    seed: int = 0
    use_kv_cache: bool = True


class GenerateResponse(BaseModel):
    generated_text: str
    generated_token_ids: list[int]
    prompt_tokens: int
    generated_tokens: int
    prefill_latency_ms: float
    decode_latency_ms: float
    total_latency_ms: float
    time_to_first_token_ms: float
    tokens_per_second: float
    model_id: str
    checkpoint_step: int
    device_platform: str
    precision: str
    cache_enabled: bool
    request_id: str


class ModelInfo(BaseModel):
    model_id: str
    version: int
    checkpoint_path: str
    training_step: int
    parameter_count: int
    max_sequence_length: int
    precision: str
    status: str
    validation_loss: float | None
    created_at: str
    status_detail: str | None = None


class ModelListResponse(BaseModel):
    models: list[ModelInfo]


class LoadRequest(BaseModel):
    checkpoint_path: str = Field(min_length=1)
    model_id: str | None = Field(
        default=None, description="Defaults to the checkpoint run directory name"
    )


class LoadResponse(BaseModel):
    model_id: str
    status: str
    load_seconds: float
    warmup_seconds: float | None


class UnloadRequest(BaseModel):
    model_id: str


class StatusResponse(BaseModel):
    status: str
    detail: str | None = None


class ErrorResponse(BaseModel):
    error: str
    request_id: str | None = None
