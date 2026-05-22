"""Process-wide lazy singleton for the AceStepHandler with ScragVAE + LoRA.

Loaded once on the first call to ``get_handler()``.  Subsequent calls return
the same object without reloading.  All configuration is read from environment
variables with the snapshot's defaults.

Environment variables:
    COVER_API_CONFIG_PATH   DiT checkpoint name (default: acestep-v15-xl-sft).
    COVER_API_LORA_PATH     LoRA directory (default: checkpoints/lora_sliders/digital-acoustic).
    COVER_API_SCRAG_REPO    HuggingFace repo for ScragVAE (default: scragnog/Ace-Step-1.5-ScragVAE).
    COVER_API_HF_CACHE_DIR  Optional HF cache directory for hf_hub_download.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

from loguru import logger

_handler: Optional["AceStepHandler"] = None  # type: ignore[name-defined]  # noqa: F821
_lock = threading.Lock()

_DEFAULT_CONFIG_PATH = "acestep-v15-xl-sft"
_DEFAULT_LORA_PATH = "checkpoints/lora_sliders/digital-acoustic"
_DEFAULT_SCRAG_REPO = "scragnog/Ace-Step-1.5-ScragVAE"
_SCRAG_FILENAME = "diffusion_pytorch_model.safetensors"


def _load_scrag_vae(handler: "AceStepHandler") -> None:  # type: ignore[name-defined]  # noqa: F821
    """Download ScragVAE weights from HuggingFace and patch the VAE decoder in-place.

    Args:
        handler: Initialized AceStepHandler whose ``vae`` attribute will be patched.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    repo_id = os.environ.get("COVER_API_SCRAG_REPO", _DEFAULT_SCRAG_REPO)
    cache_dir = os.environ.get("COVER_API_HF_CACHE_DIR") or None

    logger.info(f"[handler_setup] Downloading ScragVAE from {repo_id} ...")
    scrag_path = hf_hub_download(
        repo_id,
        _SCRAG_FILENAME,
        cache_dir=cache_dir,
    )
    scrag_weights = load_file(scrag_path)
    decoder_keys = {k: v for k, v in scrag_weights.items() if k.startswith("decoder.")}
    handler.vae.load_state_dict(decoder_keys, strict=False)
    logger.info(f"[handler_setup] ScragVAE loaded: {len(decoder_keys)} decoder weights from {scrag_path}")


def _load_lora(handler: "AceStepHandler") -> None:  # type: ignore[name-defined]  # noqa: F821
    """Load the digital-acoustic concept slider LoRA into the handler.

    Args:
        handler: Initialized AceStepHandler to inject the LoRA into.

    Raises:
        RuntimeError: If the LoRA directory does not exist.
    """
    lora_path = os.environ.get("COVER_API_LORA_PATH", _DEFAULT_LORA_PATH)
    if not os.path.exists(lora_path):
        raise RuntimeError(
            f"[handler_setup] LoRA path not found: {lora_path}. "
            "Set COVER_API_LORA_PATH or place the adapter at the default location."
        )
    result = handler.add_lora(lora_path)
    logger.info(f"[handler_setup] LoRA add result: {result}")


def _build_handler() -> "AceStepHandler":  # type: ignore[name-defined]  # noqa: F821
    """Instantiate and fully initialize the AceStepHandler with ScragVAE + LoRA.

    Returns:
        Fully initialized AceStepHandler ready for ``service_generate`` calls.

    Raises:
        RuntimeError: If model initialization fails.
    """
    from acestep.handler import AceStepHandler

    config_path = os.environ.get("COVER_API_CONFIG_PATH", _DEFAULT_CONFIG_PATH)
    logger.info(f"[handler_setup] Initializing AceStepHandler (config={config_path}) ...")

    handler = AceStepHandler()
    result, ok = handler.initialize_service(
        project_root=".",
        config_path=config_path,
    )
    if not ok:
        raise RuntimeError(f"[handler_setup] Model initialization failed:\n{result}")
    logger.info(f"[handler_setup] Model initialized: {result}")

    _load_scrag_vae(handler)
    _load_lora(handler)

    return handler


def get_handler() -> "AceStepHandler":  # type: ignore[name-defined]  # noqa: F821
    """Return the process-wide AceStepHandler, initializing it on first call.

    Thread-safe via a module-level lock.  Subsequent calls are lock-free reads.

    Returns:
        The fully initialized AceStepHandler singleton.

    Raises:
        RuntimeError: On first call, if initialization fails.
    """
    global _handler
    if _handler is not None:
        return _handler
    with _lock:
        if _handler is None:
            _handler = _build_handler()
    return _handler
