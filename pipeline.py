"""Cover generation pipeline: one job from stems to FLAC output.

Orchestrates hint extraction (optional), LoRA scale, ``service_generate``,
VAE decode, normalization, and FLAC write.  Has no Flask or job-store
dependencies — purely data-in / file-out.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from loguru import logger


@dataclasses.dataclass
class CoverParams:
    """All user-controlled generation parameters for a single cover run.

    Attributes:
        captions: Style / genre prompt string.
        lyrics: Lyric text (default ``[Instrumental]``).
        bpm: Beats per minute for the track meta.
        keyscale: Key and scale string passed to the model meta (e.g. ``"C major"``).
            Empty string means unspecified.
        time_signature: Time signature string (e.g. ``"4/4"``).
        guidance_scale: Classifier-free guidance scale.
        infer_steps: Number of diffusion inference steps.
        shift: Flow ODE shift value.
        cover_noise_strength: Amount of noise added during cover repaint.
        audio_cover_strength: Audio conditioning strength (0–1).
        lora_scale: digital-acoustic LoRA influence (0–1).
        seed: Optional RNG seed for reproducible results.
        model: DiT checkpoint name (``"acestep-v15-xl-sft"`` or ``"acestep-v15-turbo"``).
        no_fsq: When ``True``, uses ``task_type="cover-nofsq"`` which skips
            FSQ-quantized source conditioning for more creative output.
    """

    captions: str = "modern electronic pop, bright synth arpeggios, punchy drums, deep sub bass"
    lyrics: str = "[Instrumental]"
    bpm: int = 100
    keyscale: str = ""
    time_signature: str = "4/4"
    guidance_scale: float = 12.0
    infer_steps: int = 65
    shift: float = 6.0
    cover_noise_strength: float = 0.15
    audio_cover_strength: float = 1.0
    lora_scale: float = 0.7
    seed: Optional[int] = None
    model: str = "acestep-v15-xl-sft"
    no_fsq: bool = False


def _load_instrumental(instrumental_path: str) -> tuple[torch.Tensor, float]:
    """Read an instrumental stem, ensure stereo, resample to 48 kHz.

    Args:
        instrumental_path: Path to the instrumental audio file.

    Returns:
        Tuple of ``(target_wavs, duration_seconds)``.
        ``target_wavs`` has shape ``[1, 2, samples]``.
    """
    audio, sr = sf.read(instrumental_path)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)
    if sr != 48000:
        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=48000).T
        sr = 48000
    duration = len(audio) / sr
    target_wavs = torch.tensor(audio.T, dtype=torch.float32).unsqueeze(0)
    return target_wavs, duration


def _normalize_audio(audio_np: np.ndarray, peak: float = 0.891) -> np.ndarray:
    """Peak-normalize audio to ``peak`` (0–1 range).

    Args:
        audio_np: Audio numpy array (any shape).
        peak: Target peak amplitude (default matches the snapshot's 0.891).

    Returns:
        Normalized array of the same shape.
    """
    max_val = np.max(np.abs(audio_np))
    if max_val > 0:
        audio_np = audio_np / max_val * peak
    return audio_np


def run_cover(
    handler: object,
    params: CoverParams,
    instrumental_path: str,
    bass_path: Optional[str],
    out_path: str,
) -> str:
    """Run a full ScragVAE cover generation and write the result to disk.

    When ``bass_path`` is provided, semantic hints are extracted from the bass
    stem and injected via a monkey-patch on ``handler.model.prepare_condition``
    for the duration of the generation call.  The patch is always restored in
    a ``finally`` block regardless of success or failure.

    Args:
        handler: Fully initialized AceStepHandler.
        params: User-controlled generation parameters.
        instrumental_path: Path to the instrumental stem (cover source).
        bass_path: Optional path to bass stem for semantic hints.
        out_path: Destination path for the output FLAC file.

    Returns:
        Absolute path to the written FLAC file.

    Raises:
        KeyError: If ``service_generate`` does not return ``target_latents``.
        Exception: Any downstream model / IO errors propagate unchanged.
    """
    from acestep.cover_api.hints import extract_semantic_hints

    target_wavs, duration = _load_instrumental(instrumental_path)
    metas: dict = {"audio_duration": duration, "time_signature": params.time_signature, "bpm": params.bpm}
    if params.keyscale.strip():
        metas["keyscale"] = params.keyscale.strip()
    metas = [metas]

    hints: Optional[torch.Tensor] = None
    if bass_path is not None:
        logger.info(f"[pipeline] Extracting semantic hints from bass stem: {bass_path}")
        hints = extract_semantic_hints(handler, bass_path)
        logger.info(f"[pipeline] Hints shape: {hints.shape}")

    handler.set_lora_scale(params.lora_scale)  # type: ignore[attr-defined]
    handler.use_lora = True  # type: ignore[attr-defined]

    original_prepare = handler.model.prepare_condition  # type: ignore[attr-defined]

    def _patched_prepare(*args, **kwargs):
        kwargs["precomputed_lm_hints_25Hz"] = hints.to(
            device=handler.device,  # type: ignore[attr-defined]
            dtype=handler.dtype,  # type: ignore[attr-defined]
        )
        return original_prepare(*args, **kwargs)

    try:
        if hints is not None:
            handler.model.prepare_condition = _patched_prepare  # type: ignore[attr-defined]

        task_type = "cover-nofsq" if params.no_fsq else "cover"
        logger.info(
            f"[pipeline] Generating cover — steps={params.infer_steps}, "
            f"cfg={params.guidance_scale}, cns={params.cover_noise_strength}, "
            f"lora_scale={params.lora_scale}, task_type={task_type}, "
            f"keyscale={params.keyscale!r}"
        )
        result = handler.service_generate(  # type: ignore[attr-defined]
            captions=params.captions,
            lyrics=params.lyrics,
            target_wavs=target_wavs,
            metas=metas,
            audio_cover_strength=params.audio_cover_strength,
            guidance_scale=params.guidance_scale,
            infer_steps=params.infer_steps,
            shift=params.shift,
            cover_noise_strength=params.cover_noise_strength,
            task_type=task_type,
            infer_method="ode",
            seed=params.seed,
        )
    finally:
        if hints is not None:
            handler.model.prepare_condition = original_prepare  # type: ignore[attr-defined]

    if "target_latents" not in result:
        raise KeyError(
            f"[pipeline] service_generate did not return 'target_latents'. "
            f"Keys present: {list(result.keys())}"
        )

    latents = result["target_latents"]
    # Snapshot's shape guard: [B, frames, 64] → [B, 64, frames]
    if latents.shape[-1] == 64:
        latents = latents.movedim(-1, -2)

    with torch.no_grad():
        audio_tensor = handler.tiled_decode(latents)  # type: ignore[attr-defined]

    audio_np = audio_tensor.float().cpu().numpy().squeeze()
    audio_np = _normalize_audio(audio_np)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio_np.T, 48000)
    logger.info(f"[pipeline] Output written: {out_path}")
    return str(Path(out_path).resolve())
