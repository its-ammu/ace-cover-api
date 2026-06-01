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

# One handler per model config, lazily initialized on first use.
_handlers: dict[str, "AceStepHandler"] = {}  # type: ignore[name-defined]  # noqa: F821
_lock = threading.Lock()

ALLOWED_MODELS = ("acestep-v15-xl-sft", "acestep-v15-turbo")
_DEFAULT_CONFIG_PATH = "acestep-v15-xl-sft"
_DEFAULT_LORA_PATH = "checkpoints/lora_sliders/digital-acoustic"
_DEFAULT_LORA_REPO = "Xanthius/Ace-Step-1.5-XL-Concept-Sliders"
_DEFAULT_LORA_FILENAME = "ace-step_1-5_xl_digital-acoustic_slider.safetensors"
_DEFAULT_SCRAG_REPO = "scragnog/Ace-Step-1.5-ScragVAE"
_SCRAG_FILENAME = "diffusion_pytorch_model.safetensors"

# Base PEFT adapter_config.json template — rank and target_modules are filled
# in dynamically by inspecting the actual downloaded weights (same as ensure_slider_lora).
_PEFT_ADAPTER_CONFIG_TEMPLATE = {
    "alpha_pattern": {},
    "auto_mapping": None,
    "base_model_name_or_path": "",
    "bias": "none",
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_lora_weights": True,
    "layers_pattern": None,
    "layers_to_transform": None,
    "lora_dropout": 0.0,
    "modules_to_save": None,
    "peft_type": "LORA",
    "rank_pattern": {},
    "revision": None,
    "task_type": None,
    "use_rslora": False,
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

    if not decoder_keys:
        logger.warning("[handler_setup] ScragVAE: no decoder.* keys found — skipping")
        return

    # Cast to VAE's dtype to avoid dtype mismatch (same as generate_semantic.py)
    vae = handler.vae
    vae_dtype = next(vae.parameters()).dtype
    decoder_keys = {k: v.to(dtype=vae_dtype) for k, v in decoder_keys.items()}

    vae.load_state_dict(decoder_keys, strict=False)
    logger.info(f"[handler_setup] ScragVAE loaded: {len(decoder_keys)} decoder weights")


def _download_lora(dest_dir: Path) -> None:
    """Download and reformat the concept slider LoRA into a PEFT-loadable directory.

    Mirrors ``ensure_slider_lora`` from scripts/cover_pipeline/generate_semantic.py:
      1. Downloads raw safetensors from HuggingFace.
      2. Remaps ``diffusion_model.decoder.`` → ``base_model.model.`` on all keys.
      3. Detects rank from the first ``lora_A.weight`` tensor shape (not hardcoded).
      4. Detects target_modules by parsing key names (not hardcoded).
      5. Saves ``adapter_model.safetensors`` + ``adapter_config.json``.

    Args:
        dest_dir: Destination directory (created if absent).
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file, save_file

    repo_id = os.environ.get("COVER_API_LORA_REPO", _DEFAULT_LORA_REPO)
    filename = os.environ.get("COVER_API_LORA_FILENAME", _DEFAULT_LORA_FILENAME)
    cache_dir = os.environ.get("COVER_API_HF_CACHE_DIR") or None

    logger.info(f"[handler_setup] Downloading LoRA {filename} from {repo_id} ...")
    hf_path = hf_hub_download(repo_id, filename, cache_dir=cache_dir)

    tensors = load_file(hf_path)
    remapped = {
        k.replace("diffusion_model.decoder.", "base_model.model."): v
        for k, v in tensors.items()
    }

    # Detect rank from first lora_A tensor (same as ensure_slider_lora)
    rank = 8  # fallback default
    for k, v in remapped.items():
        if "lora_A.weight" in k:
            rank = v.shape[0]
            break

    # Detect target_modules from key names (same as ensure_slider_lora)
    modules: set[str] = set()
    for k in remapped.keys():
        parts = k.replace("base_model.model.", "").split(".lora_")[0]
        module_name = parts.split(".")[-1]
        modules.add(module_name)

    dest_dir.mkdir(parents=True, exist_ok=True)
    save_file(remapped, str(dest_dir / "adapter_model.safetensors"))
    logger.info(
        f"[handler_setup] Saved remapped LoRA weights "
        f"({len(remapped)} keys, rank={rank}, targets={sorted(modules)})"
    )

    adapter_config = dest_dir / "adapter_config.json"
    if not adapter_config.exists():
        config = dict(_PEFT_ADAPTER_CONFIG_TEMPLATE)
        config["r"] = rank
        config["lora_alpha"] = rank  # alpha == rank (matches ensure_slider_lora)
        config["target_modules"] = sorted(modules)
        adapter_config.write_text(json.dumps(config, indent=2))
        logger.info(f"[handler_setup] Wrote adapter_config.json (rank={rank})")

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

    # Trigger: adapter_config.json missing (same check as ensure_slider_lora).
    # This means a partially-downloaded directory re-triggers correctly.
    if not (lora_dir / "adapter_config.json").exists():
        logger.info(
            f"[handler_setup] adapter_config.json not found in {lora_dir}. "
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


def _build_handler(config_path: str) -> "AceStepHandler":  # type: ignore[name-defined]  # noqa: F821
    """Instantiate and fully initialize an AceStepHandler with ScragVAE + LoRA.

    Args:
        config_path: DiT checkpoint name (e.g. ``"acestep-v15-xl-sft"``).

    Returns:
        Fully initialized AceStepHandler ready for ``service_generate`` calls.

    Raises:
        RuntimeError: If model initialization fails.
    """
    from acestep.handler import AceStepHandler

    logger.info(f"[handler_setup] Initializing AceStepHandler (config={config_path}) ...")

    handler = AceStepHandler()
    result, ok = handler.initialize_service(project_root=".", config_path=config_path)
    if not ok:
        raise RuntimeError(f"[handler_setup] Model initialization failed:\n{result}")
    logger.info(f"[handler_setup] Model initialized: {result}")

    _load_scrag_vae(handler)
    _load_lora(handler)

    return handler


def get_handler(config_path: Optional[str] = None) -> "AceStepHandler":  # type: ignore[name-defined]  # noqa: F821
    """Return a per-model AceStepHandler, initializing it on first use.

    Each distinct ``config_path`` gets its own singleton, loaded once and
    reused for all subsequent calls with the same model.  Thread-safe via a
    module-level lock.

    Args:
        config_path: DiT checkpoint name.  Defaults to the env var
            ``COVER_API_CONFIG_PATH`` or ``"acestep-v15-xl-sft"``.

    Returns:
        The fully initialized AceStepHandler for that model.

    Raises:
        ValueError: If ``config_path`` is not in ``ALLOWED_MODELS``.
        RuntimeError: On first call for a model, if initialization fails.
    """
    if config_path is None:
        config_path = os.environ.get("COVER_API_CONFIG_PATH", _DEFAULT_CONFIG_PATH)

    if config_path not in ALLOWED_MODELS:
        raise ValueError(
            f"[handler_setup] Unknown model '{config_path}'. "
            f"Allowed: {ALLOWED_MODELS}"
        )

    if config_path in _handlers:
        return _handlers[config_path]

    with _lock:
        if config_path not in _handlers:
            _handlers[config_path] = _build_handler(config_path)

    return _handlers[config_path]
