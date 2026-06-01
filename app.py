"""Flask application factory for the Cover API.

Usage::

    from acestep.cover_api.app import create_app
    app = create_app()
    app.run(host="0.0.0.0", port=7861, threaded=True)

The factory:
1. Creates the Flask app with the templates folder from this package.
2. Builds the in-memory JobStore.
3. Registers all HTTP routes via the Blueprint.
4. Starts the single background worker thread that processes jobs.

The worker calls :func:`acestep.cover_api.pipeline.run_cover` for each job,
using the process-wide handler returned by
:func:`acestep.cover_api.handler_setup.get_handler`.
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask
from loguru import logger

from acestep.cover_api.jobs import JobStore, start_worker
from acestep.cover_api.routes import register_routes


def _make_runner(store: JobStore):
    """Return a runner callable that the worker thread will invoke per job.

    The runner is closed over ``store`` but is otherwise stateless — it calls
    ``get_handler()`` lazily so the first job triggers model initialization.

    Args:
        store: The shared JobStore (used only for output-dir resolution here).

    Returns:
        Callable ``(job) -> output_path``.
    """

    def runner(job):
        from acestep.cover_api import handler_setup, pipeline

        output_dir = os.environ.get("COVER_API_OUTPUT_DIR", "data/output/cover_api")
        out_path = os.path.join(output_dir, f"{job.job_id}.flac")

        params = pipeline.CoverParams(**job.params)
        handler = handler_setup.get_handler(config_path=params.model)

        return pipeline.run_cover(
            handler=handler,
            params=params,
            instrumental_path=job.instrumental_path,
            bass_path=job.bass_path,
            out_path=out_path,
        )

    return runner


def create_app() -> Flask:
    """Create and configure the Cover API Flask application.

    Returns:
        A ready-to-run Flask application instance.
    """
    template_dir = str(Path(__file__).parent / "templates")
    app = Flask(__name__, template_folder=template_dir)

    # Allow larger uploads (stems can be 50-200 MB)
    app.config["MAX_CONTENT_LENGTH"] = int(
        os.environ.get("COVER_API_MAX_UPLOAD_MB", "512")
    ) * 1024 * 1024

    store = JobStore()
    # Public prefix = what the browser sees (ALB path rule).
    # Flask itself always receives requests at / because the ALB strips the prefix.
    public_prefix = os.environ.get("COVER_API_URL_PREFIX", "/aceapi")
    register_routes(app, store, public_prefix=public_prefix)

    runner = _make_runner(store)
    worker_thread = start_worker(store, runner)
    app.config["COVER_WORKER_THREAD"] = worker_thread

    logger.info("[app] Cover API application created, worker thread started")
    return app
