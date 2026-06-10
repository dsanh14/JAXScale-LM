# CPU serving image. Multi-stage: uv resolves/locks deps, runtime stays slim.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Layer-cache dependencies separately from source.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

FROM python:3.12-slim

RUN useradd --create-home appuser
WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    JAX_PLATFORMS=cpu \
    JAXSCALE_ARTIFACTS_DIR=/app/artifacts

EXPOSE 8000

# Mount a checkpoint under /app/artifacts and pass --checkpoint, e.g.:
#   docker run -p 8000:8000 -v $PWD/artifacts:/app/artifacts jaxscale-lm \
#     --checkpoint artifacts/checkpoints/cpu_smoke/latest --host 0.0.0.0
ENTRYPOINT ["python", "scripts/serve.py"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
