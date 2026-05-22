"""CLI entry point for the Cover API Flask server.

Usage::

    uv run --no-sync python -m acestep.cover_api.server_cli [--host HOST] [--port PORT]

Or via the project launcher::

    ./start_cover_api.sh

Environment variables (all optional):
    COVER_API_HOST              Bind host (default 0.0.0.0).
    COVER_API_PORT              Bind port (default 7861).
    COVER_API_CONFIG_PATH       DiT checkpoint (default acestep-v15-xl-sft).
    COVER_API_LORA_PATH         LoRA directory (default checkpoints/lora_sliders/digital-acoustic).
    COVER_API_SCRAG_REPO        HuggingFace repo for ScragVAE.
    COVER_API_HF_CACHE_DIR      HF model cache override.
    COVER_API_OUTPUT_DIR        Output FLAC directory (default data/output/cover_api).
    COVER_API_MAX_UPLOAD_MB     Max upload size in MB (default 512).
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    """Parse CLI arguments and start the Flask development server.

    Flask's built-in server is intentionally used here for simplicity on a
    single EC2 instance.  ``threaded=True`` lets the polling endpoints respond
    while the worker is executing a long-running generation.
    """
    parser = argparse.ArgumentParser(description="ACE-Step ScragVAE Cover API")
    parser.add_argument(
        "--host",
        default=os.getenv("COVER_API_HOST", "0.0.0.0"),
        help="Bind host (default from COVER_API_HOST or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("COVER_API_PORT", "7861")),
        help="Bind port (default from COVER_API_PORT or 7861)",
    )
    args = parser.parse_args()

    from acestep.cover_api.app import create_app

    app = create_app()
    print(f"[cover-api] Starting on http://{args.host}:{args.port}")
    print(f"[cover-api] UI available at http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
