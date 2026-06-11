"""Serve a checkpoint over HTTP with FastAPI/Uvicorn.

Example:
    uv run python scripts/serve.py \
        --checkpoint artifacts/checkpoints/cpu_smoke/latest \
        --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint to load at startup (otherwise POST /v1/models/load later)",
    )
    parser.add_argument("--config", default="configs/inference/default.yaml")
    parser.add_argument("--host", default=None, help="Override serving.host")
    parser.add_argument("--port", type=int, default=None, help="Override serving.port")
    args = parser.parse_args()

    import uvicorn

    from jaxscale_lm.config import load_config
    from jaxscale_lm.serving.app import create_app
    from jaxscale_lm.utils.logging import setup_logging

    config = load_config(args.config)
    updates = {}
    if args.host is not None:
        updates["host"] = args.host
    if args.port is not None:
        updates["port"] = args.port
    if updates:
        config = config.model_copy(update={"serving": config.serving.model_copy(update=updates)})
    setup_logging(config.logging.level, config.logging.json_format)

    app = create_app(config, initial_checkpoint=args.checkpoint)
    uvicorn.run(app, host=config.serving.host, port=config.serving.port, log_level="info")


if __name__ == "__main__":
    main()
