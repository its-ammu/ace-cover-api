"""Flask Blueprint for the Cover API HTTP layer.

Endpoints (all mounted under the configured url_prefix, default /aceapi):
    GET  /                        Serve the browser UI.
    POST /api/cover               Accept multipart upload; enqueue cover job.
    GET  /api/jobs/<job_id>       Poll job status.
    GET  /api/jobs/<job_id>/download  Stream finished FLAC to caller.
    GET  /api/health              Liveness check.

All JSON responses use the shape ``{"ok": bool, ...}``.
"""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

from flask import Blueprint, jsonify, render_template, request, send_file

from acestep.cover_api.jobs import JobStatus
from acestep.cover_api.pipeline import CoverParams

if TYPE_CHECKING:
    from acestep.cover_api.jobs import JobStore

bp = Blueprint("cover", __name__)

_ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".m4a"}


def _parse_params() -> CoverParams:
    """Parse and validate generation parameters from the current request form.

    Returns:
        Populated CoverParams with form values (falling back to defaults).

    Raises:
        ValueError: If a numeric field cannot be coerced to its expected type.
    """
    f = request.form

    def _float(key: str, default: float) -> float:
        return float(f.get(key, default))

    def _int(key: str, default: int) -> int:
        return int(f.get(key, default))

    seed_raw = f.get("seed", "")
    seed = int(seed_raw) if seed_raw.strip() else None

    return CoverParams(
        captions=f.get("captions", CoverParams.captions),
        lyrics=f.get("lyrics", CoverParams.lyrics),
        bpm=_int("bpm", CoverParams.bpm),
        time_signature=f.get("time_signature", CoverParams.time_signature),
        guidance_scale=_float("guidance_scale", CoverParams.guidance_scale),
        infer_steps=_int("infer_steps", CoverParams.infer_steps),
        shift=_float("shift", CoverParams.shift),
        cover_noise_strength=_float("cover_noise_strength", CoverParams.cover_noise_strength),
        audio_cover_strength=_float("audio_cover_strength", CoverParams.audio_cover_strength),
        lora_scale=_float("lora_scale", CoverParams.lora_scale),
        seed=seed,
    )


def _save_upload(file_obj, suffix: str) -> str:
    """Save a Werkzeug FileStorage to a named temp file.

    Args:
        file_obj: The uploaded FileStorage object.
        suffix: File extension to use (e.g. ``.wav``).

    Returns:
        Absolute path to the saved file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    file_obj.save(tmp.name)
    return tmp.name


def _safe_ext(filename: str) -> str:
    """Return the lowercased extension of *filename*, or ``.wav`` as fallback.

    Args:
        filename: Original client filename.

    Returns:
        Dot-prefixed lowercase extension string.
    """
    _, ext = os.path.splitext(filename or "")
    ext = ext.lower()
    return ext if ext in _ALLOWED_AUDIO_EXT else ".wav"


def register_routes(app, store: "JobStore", public_prefix: str = "/aceapi") -> None:
    """Register the Blueprint and inject ``store`` into the app context.

    The blueprint is always mounted at Flask root (``/``) because the ALB
    strips the path prefix before forwarding to this server.
    ``public_prefix`` is the browser-visible prefix (e.g. ``/aceapi``) used
    only for building URLs returned to clients (JS base URL, download_url).

    Args:
        app: Flask application instance.
        store: Shared JobStore for job creation and lookup.
        public_prefix: The public-facing path prefix seen by the browser.
            Does NOT affect Flask routing (ALB strips it before it arrives).
            Set ``COVER_API_URL_PREFIX`` env var to override.
    """
    app.config["COVER_STORE"] = store
    app.config["COVER_PUBLIC_PREFIX"] = public_prefix.rstrip("/")
    app.register_blueprint(bp)  # ALB strips prefix — Flask always sees /


@bp.route("/")
def index():
    """Serve the single-page browser UI."""
    from flask import current_app
    url_prefix = current_app.config.get("COVER_PUBLIC_PREFIX", "/aceapi")
    return render_template("index.html", url_prefix=url_prefix)


@bp.route("/api/health")
def health():
    """Return a simple liveness response."""
    return jsonify({"ok": True, "status": "healthy"})


@bp.route("/api/cover", methods=["POST"])
def submit_cover():
    """Accept a multipart form upload and enqueue a cover generation job.

    Form fields:
        instrumental_stem (file, required): Instrumental audio file.
        bass_stem (file, optional): Bass stem for semantic hints.
        captions, lyrics, bpm, time_signature, guidance_scale, infer_steps,
        shift, cover_noise_strength, audio_cover_strength, lora_scale,
        seed: Generation parameters (all optional, defaults match snapshot).

    Returns:
        JSON ``{"ok": true, "job_id": "<uuid>", "status": "queued"}`` or
        ``{"ok": false, "error": "<message>"}`` on validation failure.
    """
    store: JobStore = _current_store()

    if "instrumental_stem" not in request.files:
        return jsonify({"ok": False, "error": "instrumental_stem file is required"}), 400

    instr_file = request.files["instrumental_stem"]
    if not instr_file.filename:
        return jsonify({"ok": False, "error": "instrumental_stem has no filename"}), 400

    try:
        params = _parse_params()
    except (ValueError, TypeError) as exc:
        return jsonify({"ok": False, "error": f"Invalid parameter: {exc}"}), 400

    instr_path = _save_upload(instr_file, _safe_ext(instr_file.filename))

    bass_path = None
    bass_file = request.files.get("bass_stem")
    if bass_file and bass_file.filename:
        bass_path = _save_upload(bass_file, _safe_ext(bass_file.filename))

    job = store.create(
        params=vars(params),
        instrumental_path=instr_path,
        bass_path=bass_path,
    )
    return jsonify({"ok": True, "job_id": job.job_id, "status": job.status.value}), 202


@bp.route("/api/jobs/<job_id>")
def job_status(job_id: str):
    """Return the current status of a job.

    Args:
        job_id: UUID of the job to poll.

    Returns:
        JSON ``{"ok": true, "job_id": ..., "status": ..., "download_url": ...}``
        where ``download_url`` is set only when status is ``done``.
    """
    store: JobStore = _current_store()
    job = store.get(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "job not found"}), 404

    data = job.to_dict()
    data["ok"] = True
    if job.status == JobStatus.DONE:
        from flask import current_app
        prefix = current_app.config.get("COVER_PUBLIC_PREFIX", "")
        data["download_url"] = f"{prefix}/api/jobs/{job_id}/download"
    return jsonify(data)


@bp.route("/api/jobs/<job_id>/download")
def download_result(job_id: str):
    """Stream the generated FLAC file to the caller.

    Args:
        job_id: UUID of a completed job.

    Returns:
        FLAC audio bytes, or a JSON error if the job is not ready / not found.
    """
    store: JobStore = _current_store()
    job = store.get(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "job not found"}), 404
    if job.status != JobStatus.DONE:
        return jsonify({"ok": False, "error": f"job not ready (status={job.status.value})"}), 409
    if not job.output_path or not os.path.exists(job.output_path):
        return jsonify({"ok": False, "error": "output file not found on disk"}), 500

    return send_file(
        job.output_path,
        mimetype="audio/flac",
        as_attachment=True,
        download_name=f"cover_{job_id}.flac",
    )


def _current_store() -> "JobStore":
    """Retrieve the JobStore from the current Flask application config.

    Returns:
        The JobStore instance registered during app creation.
    """
    from flask import current_app

    return current_app.config["COVER_STORE"]
