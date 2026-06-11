"""FastAPI application.

Endpoints (see docs/architecture.md for the flow diagram):

- ``GET  /health``               liveness (process is up)
- ``GET  /ready``                readiness (a model is loaded AND warm)
- ``GET  /v1/models``            list registered model versions
- ``GET  /v1/models/{model_id}`` model metadata
- ``POST /v1/models/load``       load a checkpoint (warms up before READY)
- ``POST /v1/models/unload``     unload the active model
- ``POST /v1/generate``          generate text
- ``GET  /metrics``              Prometheus exposition

Error policy: validation problems -> 400 with the actual message (these are
safe, actionable strings raised by our own validators); no model -> 503;
unexpected errors -> 500 with a generic body, full traceback only in server
logs. Every request gets a UUID request id, echoed in responses and logs.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import jax
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from jaxscale_lm import __version__
from jaxscale_lm.config import Config, InferenceConfig
from jaxscale_lm.serving import metrics
from jaxscale_lm.serving.lifecycle import ModelManager
from jaxscale_lm.serving.registry import ModelRegistry
from jaxscale_lm.serving.schemas import (
    GenerateRequest,
    GenerateResponse,
    LoadRequest,
    LoadResponse,
    ModelInfo,
    ModelListResponse,
    StatusResponse,
    UnloadRequest,
)
from jaxscale_lm.utils.logging import get_logger, log_event

_logger = get_logger("api")


def create_app(config: Config, initial_checkpoint: str | None = None) -> FastAPI:
    """Build the application; optionally load a checkpoint during startup."""
    registry = ModelRegistry(Path(config.registry_path))
    manager = ModelManager(registry, config.serving)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if initial_checkpoint is not None:
            manager.load(initial_checkpoint)  # includes warmup before READY
        log_event(_logger, "server started", version=__version__)
        yield
        log_event(_logger, "server shutting down", active_model=manager.active_model_id)

    app = FastAPI(title="JAXScale-LM", version=__version__, lifespan=lifespan)

    # -- error handling -----------------------------------------------------
    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        _logger.exception(
            "unhandled error", extra={"request_id": request_id, "path": request.url.path}
        )
        metrics.REQUEST_ERRORS.labels(kind="internal").inc()
        return JSONResponse(
            status_code=500,
            content={"error": "internal server error", "request_id": request_id},
        )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        return response

    # -- health -------------------------------------------------------------
    @app.get("/health", response_model=StatusResponse)
    async def health() -> StatusResponse:
        return StatusResponse(status="ok")

    @app.get("/ready", response_model=StatusResponse)
    async def ready() -> StatusResponse:
        if not manager.ready:
            raise HTTPException(status_code=503, detail="no model loaded")
        return StatusResponse(status="ready", detail=manager.active_model_id)

    # -- models ---------------------------------------------------------------
    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models() -> ModelListResponse:
        return ModelListResponse(
            models=[ModelInfo(**_entry_fields(e)) for e in registry.list()]
        )

    @app.get("/v1/models/{model_id}", response_model=ModelInfo)
    async def get_model(model_id: str) -> ModelInfo:
        try:
            return ModelInfo(**_entry_fields(registry.get(model_id)))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/models/load", response_model=LoadResponse)
    async def load_model(body: LoadRequest) -> LoadResponse:
        try:
            model_id, load_s, warmup_s = manager.load(body.checkpoint_path, body.model_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return LoadResponse(
            model_id=model_id, status="READY", load_seconds=load_s, warmup_seconds=warmup_s
        )

    @app.post("/v1/models/unload", response_model=StatusResponse)
    async def unload_model(body: UnloadRequest) -> StatusResponse:
        try:
            manager.unload(body.model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return StatusResponse(status="UNLOADED", detail=body.model_id)

    # -- generation -----------------------------------------------------------
    @app.post("/v1/generate", response_model=GenerateResponse)
    async def generate(body: GenerateRequest, request: Request) -> GenerateResponse:
        request_id: str = request.state.request_id
        if not manager.ready:
            metrics.REQUEST_ERRORS.labels(kind="no_model").inc()
            raise HTTPException(status_code=503, detail="no model loaded")
        if body.max_new_tokens > config.serving.max_new_tokens_limit:
            metrics.REQUEST_ERRORS.labels(kind="validation").inc()
            raise HTTPException(
                status_code=400,
                detail=(
                    f"max_new_tokens ({body.max_new_tokens}) exceeds the server limit "
                    f"({config.serving.max_new_tokens_limit})"
                ),
            )
        options = InferenceConfig(
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            top_p=body.top_p,
            do_sample=body.do_sample,
            seed=body.seed,
            use_kv_cache=body.use_kv_cache,
        )
        log_event(
            _logger,
            "generate request",
            request_id=request_id,
            prompt_chars=len(body.prompt),  # never the prompt text itself
            max_new_tokens=body.max_new_tokens,
            cache=body.use_kv_cache,
        )
        metrics.REQUESTS.labels(cache_enabled=str(body.use_kv_cache).lower()).inc()
        metrics.ACTIVE_REQUESTS.inc()
        start = time.perf_counter()
        try:
            prompt_token_count = len(manager.engine.tokenizer.encode(body.prompt))
            if prompt_token_count > config.serving.max_prompt_tokens:
                metrics.REQUEST_ERRORS.labels(kind="validation").inc()
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"prompt is {prompt_token_count} tokens; server limit is "
                        f"{config.serving.max_prompt_tokens}"
                    ),
                )
            try:
                result = manager.generate(body.prompt, options)
            except ValueError as exc:
                metrics.REQUEST_ERRORS.labels(kind="validation").inc()
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            metrics.ACTIVE_REQUESTS.dec()
        elapsed = time.perf_counter() - start

        metrics.REQUEST_LATENCY.observe(elapsed)
        metrics.PREFILL_LATENCY.observe(result.timing.prefill_s)
        metrics.DECODE_LATENCY.observe(result.timing.decode_s)
        metrics.PROMPT_TOKENS.inc(result.prompt_tokens)
        metrics.GENERATED_TOKENS.inc(result.generated_tokens)
        metrics.TOKENS_PER_SECOND.set(result.tokens_per_second)
        log_event(
            _logger,
            "generate complete",
            request_id=request_id,
            generated_tokens=result.generated_tokens,
            total_ms=round(elapsed * 1000, 1),
        )
        return GenerateResponse(
            generated_text=result.generated_text,
            generated_token_ids=result.generated_token_ids,
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.generated_tokens,
            prefill_latency_ms=result.timing.prefill_s * 1000,
            decode_latency_ms=result.timing.decode_s * 1000,
            total_latency_ms=elapsed * 1000,
            time_to_first_token_ms=result.time_to_first_token_s * 1000,
            tokens_per_second=result.tokens_per_second,
            model_id=manager.active_model_id or "unknown",
            checkpoint_step=result.checkpoint_step,
            device_platform=jax.default_backend(),
            precision=manager.engine.model_config.compute_dtype,
            cache_enabled=result.cache_enabled,
            request_id=request_id,
        )

    # -- metrics ----------------------------------------------------------
    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        return Response(content=metrics.render(), media_type="text/plain; version=0.0.4")

    return app


def _entry_fields(entry) -> dict:
    data = entry.to_dict()
    data.pop("model_config", None)
    data.pop("tokenizer_path", None)
    return {**data, "tokenizer_path": entry.tokenizer_path}
