"""Load and resolve Whisper configuration from defaults, files, env, and CLI."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODELS_ROOT = "~/.cache/whisper-models"
DEFAULT_MODEL_ID = "mlx-community/whisper-large-v3-turbo"

CONFIG_FILENAMES = ("config.yaml",)
USER_CONFIG = Path("~/.config/local-transcription-service/config.yaml")


@dataclass
class WhisperConfig:
    models_root: Path
    model_path: Path | None
    model_id: str
    language: str | None


def _expand(path: str | Path) -> Path:
    return Path(str(path)).expanduser().resolve()


def _default_dict() -> dict[str, Any]:
    return {
        "whisper": {
            "models_root": DEFAULT_MODELS_ROOT,
            "model_path": "",
            "model_id": DEFAULT_MODEL_ID,
            "language": None,
        }
    }


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _find_project_config(start: Path | None = None) -> Path | None:
    cwd = (start or Path.cwd()).resolve()
    for directory in (cwd, *cwd.parents):
        for name in CONFIG_FILENAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def _env_overlay() -> dict[str, Any]:
    whisper: dict[str, Any] = {}
    if value := os.environ.get("WHISPER_MODELS_ROOT"):
        whisper["models_root"] = value
    if value := os.environ.get("WHISPER_MODEL_PATH"):
        whisper["model_path"] = value
    if value := os.environ.get("WHISPER_MODEL_ID"):
        whisper["model_id"] = value
    if "WHISPER_LANGUAGE" in os.environ:
        lang = os.environ.get("WHISPER_LANGUAGE") or None
        whisper["language"] = lang if lang and lang.lower() not in ("", "null", "none", "auto") else None
    return {"whisper": whisper} if whisper else {}


def load_raw_config() -> dict[str, Any]:
    """Merge defaults ← user config ← project config ← env."""
    cfg = _default_dict()
    user_path = USER_CONFIG.expanduser()
    cfg = _deep_merge(cfg, _load_yaml(user_path))
    project = _find_project_config()
    if project is not None:
        cfg = _deep_merge(cfg, _load_yaml(project))
    cfg = _deep_merge(cfg, _env_overlay())
    return cfg


def resolve_whisper_config(
    *,
    models_root: str | None = None,
    model_path: str | None = None,
    model_id: str | None = None,
    language: str | None = None,
    language_explicit: bool = False,
) -> WhisperConfig:
    """Build final WhisperConfig. CLI kwargs override file/env when provided."""
    raw = load_raw_config()["whisper"]

    root = models_root if models_root is not None else raw.get("models_root", DEFAULT_MODELS_ROOT)
    path_raw = model_path if model_path is not None else (raw.get("model_path") or "")
    mid = model_id if model_id is not None else raw.get("model_id", DEFAULT_MODEL_ID)

    if language_explicit:
        lang = language
    else:
        lang = raw.get("language")
        if lang is not None and str(lang).lower() in ("", "null", "none", "auto"):
            lang = None

    local: Path | None = None
    if str(path_raw).strip():
        local = _expand(path_raw)

    return WhisperConfig(
        models_root=_expand(root),
        model_path=local,
        model_id=str(mid),
        language=str(lang) if lang else None,
    )


WEIGHT_FILENAMES = ("weights.safetensors", "weights.npz")


def resolve_model_ref(cfg: WhisperConfig) -> str:
    """Use the local MLX bundle if present, else the HuggingFace model id.

    An MLX bundle is a directory holding ``config.json`` beside a weights file.
    """
    local = cfg.model_path
    if local is not None and local.is_dir() and (local / "config.json").exists():
        if any((local / name).exists() for name in WEIGHT_FILENAMES):
            return str(local)
    return cfg.model_id
