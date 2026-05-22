"""Process-wide lazy singleton for the AceStepHandler with ScragVAE + optional LoRA.

Loaded once on the first call to ``get_handler()``.  Subsequent calls return
the same object without reloading.  All configuration is read from environment
variables with the snapshot's defaults.

Environment variables:
    COVER_API_CONFIG_PATH   DiT checkpoint name (default: acestep-v15-xl-sft).
    COVER_API_LORA_PATH     Local LoRA directory.  If absent the LoRA is
                            auto-downloaded from COVER_API_LORA_REPO.
                            Set to empty string ("") to skip LoRA entirely.
    COVER_API_LORA_REPO     HuggingFace repo for the concept slider LoRA
                            (default: Xanthius/Ace-Step-1.5-XL-Concept-Sliders).
    COVER_API_LORA_FILENAME Filename inside the HF repo to download
                            (default: ace-step_1-5_xl_digital-acoustic_slider.safetensors).
    COVER_API_SCRAG_REPO    HuggingFace repo for ScragVAE
                            (default: scragnog/Ace-Step-1.5-ScragVAE).
    COVER_API_HF_CACHE_DIR  Optional HF cache directory for hf_hub_download.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

from loguru import logger

_handler: Optional["AceStepHandler"] = None  # type: ignore[name-defined]  # noqa: F821
_lock = threading.Lock()

_DEFAULT_CONFIG_PATH = "acestep-v15-xl-sft"
_DEFAULT_LORA_PATH = "checkpoints/lora_sliders/digital-acoustic"
_DEFAULT_LORA_REPO = "Xanthius/Ace-Step-1.5-XL-Concept-Sliders"
_DEFAULT_LORA_FILENAME = "ace-step_1-5_xl_digital-acoustic_slider.safetensors"
_DEFAULT_SCRAG_REPO = "scragnog/Ace-Step-1.5-ScragVAE"
_SCRAG_FILENAME = "diffusion_pytorch_model.safetensors"

# Minimal PEFT adapter_config.json required by handler.add_lora().
# The concept slider LoRAs from Xanthius ship without one, so we create it.
_PEFT_ADAPTER_CONFIG = {
    "peft_type": "LORA",
    "task_type": "FEATURE_EXTRACTION",
    "base_model_name_or_path": "ACE-Step/acestep-v15-xl-base",
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
    "bias": "none",
    "inference_mode": True,
}


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
    scrag_path = hf_hub_download(repo_id, _SCRAG_FILENAME, cache_dir=cache_dir)
    scrag_weights = load_file(scrag_path)
    decoder_keys = {k: v for k, v in scrag_weights.items() if k.startswith("decoder.")}
    handler.vae.load_state_dict(decoder_keys, strict=False)
    logger.info(f"[handler_setup] ScragVAE loaded: {len(decoder_keys)} decoder weights")


def _download_lora(dest_dir: Path) -> None:
    """Download the concept slider LoRA from HuggingFace into ``dest_dir``.

    The Xanthius concept slider repo ships only raw ``.safetensors`` files with
    no ``adapter_config.json``.  This function downloads the weights and writes
    a minimal PEFT config alongside them so ``handler.add_lora()`` can load the
    directory correctly.

    Args:
        dest_dir: Destination directory (created if absent).
    """
    from huggingface_hub import hf_hub_download

    repo_id = os.environ.get("COVER_API_LORA_REPO", _DEFAULT_LORA_REPO)
    filename = os.environ.get("COVER_API_LORA_FILENAME", _DEFAULT_LORA_FILENAME)
    cache_dir = os.environ.get("COVER_API_HF_CACHE_DIR") or None

    logger.info(f"[handler_setup] Downloading LoRA {filename} from {repo_id} ...")
    hf_path = hf_hub_download(repo_id, filename, cache_dir=cache_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)
    adapter_weights = dest_dir / "adapter_model.safetensors"
    adapter_config = dest_dir / "adapter_config.json"

    # Copy the weights file into the PEFT directory layout
    import shutil
    shutil.copy2(hf_path, str(adapter_weights))

    if not adapter_config.exists():
        with open(adapter_config, "w") as fh:
            json.dump(_PEFT_ADAPTER_CONFIG, fh, indent=2)
        logger.info(f"[handler_setup] Wrote minimal adapter_config.json to {adapter_config}")

    logger.info(f"[handler_setup] LoRA ready at {dest_dir}")


def _load_lora(handler: "AceStepHandler") -> None:  # type: ignore[name-defined]  # noqa: F821
    """Load the digital-acoustic concept slider LoRA into the handler.

    If the configured path does not exist, the LoRA is auto-downloaded from
    HuggingFace.  Set ``COVER_API_LORA_PATH=""`` to skip LoRA entirely.

    Args:
        handler: Initialized AceStepHandler to inject the LoRA into.
    """
    lora_path_env = os.environ.get("COVER_API_LORA_PATH", _DEFAULT_LORA_PATH)

    if lora_path_env == "":
        logger.warning("[handler_setup] COVER_API_LORA_PATH is empty — skipping LoRA load.")
        handler.use_lora = False
        return

    lora_dir = Path(lora_path_env)

    if not lora_dir.exists():
        logger.info(
            f"[handler_setup] LoRA path not found locally ({lora_dir}). "
            "Auto-downloading from HuggingFace ..."
        )
        _download_lora(lora_dir)

    result = handler.add_lora(str(lora_dir))
    logger.info(f"[handler_setup] LoRA add result: {result}")

    if result.startswith("❌"):
        logger.warning(
            f"[handler_setup] LoRA failed to load: {result}. "
            "Continuing without LoRA — ScragVAE decoder is still active."
        )
        handler.use_lora = False


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
    result, ok = handler.initialize_service(project_root=".", config_path=config_path)
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
