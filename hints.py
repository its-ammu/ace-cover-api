"""Semantic hint extraction from a bass (or any reference) audio stem.

This is a direct port of ``extract_semantic_hints`` from the reproduce_final_scrag
snapshot.  Changes from the original:
- Uses ``handler.device`` and ``handler.dtype`` instead of hardcoded ``cuda:0`` /
  ``bfloat16``.
- Accepts ``handler`` as the first argument so tests can inject a mock.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import torch
from loguru import logger


def extract_semantic_hints(handler: object, audio_path: str) -> torch.Tensor:
    """Extract 25 Hz semantic hints from a reference audio file.

    Flow: read audio → mono→stereo → optional resample to 48 kHz →
    VAE encode → transpose → tokenizer → quantize → detokenize → 25 Hz hints.

    Args:
        handler: Initialized AceStepHandler providing ``tiled_encode``,
            ``silence_latent``, ``model.tokenize``, ``model.detokenize``,
            ``device``, and ``_get_vae_dtype``.
        audio_path: Path to the source audio file (any sample rate).

    Returns:
        Tensor of shape ``[1, T_25hz, hidden_dim]`` with continuous LM hints.
    """
    audio_data, sr = sf.read(audio_path)
    if audio_data.ndim == 1:
        audio_data = np.stack([audio_data, audio_data], axis=-1)

    audio_tensor = torch.tensor(audio_data.T, dtype=torch.float32)
    if sr != 48000:
        import torchaudio
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 48000)

    if audio_tensor.shape[0] == 1:
        audio_tensor = audio_tensor.repeat(2, 1)
    audio_tensor = audio_tensor.unsqueeze(0)  # [1, 2, samples]

    device = handler.device  # type: ignore[attr-defined]
    dtype = handler._get_vae_dtype()  # type: ignore[attr-defined]

    audio_tensor = audio_tensor.to(device=device, dtype=dtype)
    with torch.no_grad():
        latents = handler.tiled_encode(audio_tensor)  # type: ignore[attr-defined]

    # [batch, channels, frames] → [batch, frames, channels]
    latents_transposed = latents.movedim(-1, -2)

    tokenizer = handler.model.tokenizer  # type: ignore[attr-defined]
    model_device = next(tokenizer.parameters()).device
    model_dtype = next(tokenizer.parameters()).dtype
    latents_transposed = latents_transposed.to(device=model_device, dtype=model_dtype)

    with torch.no_grad():
        silence_latent = handler.silence_latent.to(device=model_device, dtype=model_dtype)  # type: ignore[attr-defined]
        if silence_latent.shape[1] < latents_transposed.shape[1]:
            repeats = (latents_transposed.shape[1] // silence_latent.shape[1]) + 1
            silence_latent = silence_latent.repeat(1, repeats, 1)[:, : latents_transposed.shape[1], :]

        attention_mask = torch.ones(
            latents_transposed.shape[0],
            latents_transposed.shape[1],
            device=model_device,
        )
        quantized, _indices, _mask = handler.model.tokenize(  # type: ignore[attr-defined]
            latents_transposed, silence_latent, attention_mask
        )

    with torch.no_grad():
        lm_hints = handler.model.detokenize(quantized)  # type: ignore[attr-defined]

    logger.info(f"[hints] Extracted semantic hints shape: {lm_hints.shape}")
    return lm_hints
