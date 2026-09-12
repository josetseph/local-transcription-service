"""Load and resolve Whisper configuration from defaults, files, env, and CLI."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODELS_ROOT = "~/.cache/transcribe-models"
# Per-engine defaults: an MLX repo id handed to faster-whisper (or vice versa)
# fails confusingly, so the default follows whichever engine is selected.
DEFAULT_MODEL_ID_MLX = "mlx-community/whisper-large-v3-mlx"
DEFAULT_MODEL_ID_QWEN = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_MODEL_ID_FASTER_WHISPER = "Systran/faster-whisper-large-v3"
DEFAULT_MODEL_ID = DEFAULT_MODEL_ID_MLX  # back-compat alias

ENGINE_MLX = "mlx"
ENGINE_FASTER_WHISPER = "faster-whisper"
ENGINE_QWEN = "qwen"
ENGINES = (ENGINE_QWEN, ENGINE_MLX, ENGINE_FASTER_WHISPER)
DEFAULT_ENGINE = "auto"
DEFAULT_DEVICE = "auto"
DEFAULT_COMPUTE_TYPE = "int8"

MLX_WEIGHTS = ("weights.safetensors", "weights.npz")
CT2_WEIGHTS = ("model.bin",)
DEFAULT_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"
DEFAULT_DIARIZATION_STEP = 2.0

CONFIG_FILENAMES = ("config.yaml",)
USER_CONFIG = Path("~/.config/local-transcription-service/config.yaml")


@dataclass
class WhisperConfig:
    models_root: Path
    model_path: Path | None
    model_id: str | None
    language: str | None
    engine: str = DEFAULT_ENGINE
    device: str = DEFAULT_DEVICE
    compute_type: str = DEFAULT_COMPUTE_TYPE
    # Domain terms biasing recognition (qwen only; other engines ignore it).
    context: str = ""


def _expand(path: str | Path) -> Path:
    return Path(str(path)).expanduser().resolve()


def _default_dict() -> dict[str, Any]:
    return {
        "whisper": {
            "models_root": DEFAULT_MODELS_ROOT,
            "model_path": "",
            # null means "pick the default for the selected engine"
            "model_id": None,
            "language": None,
            "engine": DEFAULT_ENGINE,
            "context": "",
            "device": DEFAULT_DEVICE,
            "compute_type": DEFAULT_COMPUTE_TYPE,
        },
        "diarization": {
            "model_id": DEFAULT_DIARIZATION_MODEL_ID,
            "step": DEFAULT_DIARIZATION_STEP,
            "token": None,
            "speakers": None,
        },
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
    if value := os.environ.get("WHISPER_ENGINE"):
        whisper["engine"] = value
    if value := os.environ.get("WHISPER_DEVICE"):
        whisper["device"] = value
    if value := os.environ.get("WHISPER_COMPUTE_TYPE"):
        whisper["compute_type"] = value
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
    engine: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    context: str | None = None,
) -> WhisperConfig:
    """Build final WhisperConfig. CLI kwargs override file/env when provided."""
    raw = load_raw_config()["whisper"]

    root = models_root if models_root is not None else raw.get("models_root", DEFAULT_MODELS_ROOT)
    path_raw = model_path if model_path is not None else (raw.get("model_path") or "")
    mid = model_id if model_id is not None else raw.get("model_id")
    eng = engine if engine is not None else raw.get("engine", DEFAULT_ENGINE)
    dev = device if device is not None else raw.get("device", DEFAULT_DEVICE)
    ctype = compute_type if compute_type is not None else raw.get("compute_type", DEFAULT_COMPUTE_TYPE)
    ctx = context if context is not None else (raw.get("context") or "")

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
        model_id=str(mid) if mid else None,
        language=str(lang) if lang else None,
        engine=str(eng),
        device=str(dev),
        compute_type=str(ctype),
        context=str(ctx),
    )


def is_mlx_bundle(path: Path) -> bool:
    """An MLX model folder: config.json beside a weights file."""
    return (
        path.is_dir()
        and (path / "config.json").exists()
        and any((path / name).exists() for name in MLX_WEIGHTS)
    )


def is_qwen_bundle(path: Path) -> bool:
    """A Qwen3-ASR folder: HF layout with a Qwen3-ASR architecture in config.json."""
    cfg = path / "config.json"
    if not (path.is_dir() and cfg.exists()):
        return False
    if not any(path.glob("*.safetensors")):
        return False
    try:
        import json

        text = json.dumps(json.loads(cfg.read_text(encoding="utf-8"))).lower()
    except (OSError, ValueError):
        return False
    return "qwen3asr" in text.replace("_", "").replace("-", "")


def is_ct2_bundle(path: Path) -> bool:
    """A CTranslate2 (faster-whisper) model folder."""
    return path.is_dir() and any((path / name).exists() for name in CT2_WEIGHTS)


def _installed(module: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def is_apple_silicon() -> bool:
    import platform

    return platform.system() == "Darwin" and platform.machine() == "arm64"


def resolve_engine(cfg: WhisperConfig) -> str:
    """Pick the backend: explicit choice, else the local model's format, else platform.

    In auto mode only an *installed* engine is chosen — a local MLX folder on a
    Linux box should not select a backend that cannot run there. An explicit
    --engine is always honoured, so the caller gets a precise install error
    rather than a silent substitution.
    """
    if cfg.engine and cfg.engine != "auto":
        if cfg.engine not in ENGINES:
            raise ValueError(
                f"Unknown engine {cfg.engine!r}; choose from {', '.join(ENGINES)} or 'auto'."
            )
        return cfg.engine

    available = {
        ENGINE_QWEN: _installed("mlx_qwen3_asr"),
        ENGINE_MLX: _installed("mlx_whisper"),
        ENGINE_FASTER_WHISPER: _installed("faster_whisper"),
    }

    # A local model folder states unambiguously which runtime can read it.
    if cfg.model_path is not None:
        if is_qwen_bundle(cfg.model_path) and available[ENGINE_QWEN]:
            return ENGINE_QWEN
        if is_mlx_bundle(cfg.model_path) and available[ENGINE_MLX]:
            return ENGINE_MLX
        if is_ct2_bundle(cfg.model_path) and available[ENGINE_FASTER_WHISPER]:
            return ENGINE_FASTER_WHISPER

    preference = (
        (ENGINE_QWEN, ENGINE_MLX, ENGINE_FASTER_WHISPER)
        if is_apple_silicon()
        else (ENGINE_FASTER_WHISPER, ENGINE_MLX)
    )
    for engine in preference:
        if available[engine]:
            return engine
    # Nothing installed: name the engine this machine ought to use, so the
    # missing-dependency error that follows points at the right package.
    return preference[0]


def default_model_id(engine: str) -> str:
    return {
        ENGINE_QWEN: DEFAULT_MODEL_ID_QWEN,
        ENGINE_MLX: DEFAULT_MODEL_ID_MLX,
    }.get(engine, DEFAULT_MODEL_ID_FASTER_WHISPER)


def resolve_model_ref(cfg: WhisperConfig, engine: str | None = None) -> tuple[str, str | None]:
    """Return ``(model_ref, warning)`` for the given engine.

    The warning is non-empty when a configured ``model_path`` had to be ignored;
    falling back silently to a download is how a wrong path turns into a
    surprise multi-gigabyte fetch of a model the engine still cannot read.
    """
    engine = engine or resolve_engine(cfg)
    matches = {
        ENGINE_QWEN: is_qwen_bundle,
        ENGINE_MLX: is_mlx_bundle,
    }.get(engine, is_ct2_bundle)
    kind = {ENGINE_QWEN: "Qwen3-ASR", ENGINE_MLX: "MLX"}.get(engine, "CTranslate2")
    wants_mlx = engine == ENGINE_MLX

    warning = None
    if cfg.model_path is not None:
        if matches(cfg.model_path):
            return str(cfg.model_path), None
        if not cfg.model_path.exists():
            warning = f"model_path {cfg.model_path} does not exist"
        else:
            found = next(
                (
                    name
                    for name, test in (
                        ("Qwen3-ASR", is_qwen_bundle),
                        ("MLX", is_mlx_bundle),
                        ("CTranslate2", is_ct2_bundle),
                    )
                    if name != kind and test(cfg.model_path)
                ),
                None,
            )
            looks = f" (it looks like a {found} folder)" if found else ""
            warning = f"model_path {cfg.model_path} is not a {kind} model{looks}"

    if not cfg.model_id:
        return default_model_id(engine), warning

    # A pinned id from the other engine's ecosystem would download gigabytes of
    # unreadable weights, so say so rather than letting it fail after the fetch.
    lowered = cfg.model_id.lower()
    looks_mlx = "mlx" in lowered
    looks_ct2 = "faster-whisper" in lowered or lowered.startswith("systran/")
    if engine == ENGINE_QWEN:
        return cfg.model_id, warning
    if wants_mlx and looks_ct2 and not looks_mlx:
        warning = warning or f"model_id {cfg.model_id} looks like a CTranslate2 model, not MLX"
    elif not wants_mlx and looks_mlx:
        warning = warning or f"model_id {cfg.model_id} looks like an MLX model, not CTranslate2"

    return cfg.model_id, warning


def resolve_diarization_config(
    *,
    model_path: str | None = None,
    model_id: str | None = None,
    step: float | None = None,
    speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
):
    """Build a DiarizationConfig. CLI kwargs override file/env when provided."""
    from local_transcription_service.diarize import DiarizationConfig

    raw = load_raw_config().get("diarization") or {}

    token = (
        raw.get("token")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    spk = speakers if speakers is not None else raw.get("speakers")
    lo = min_speakers if min_speakers is not None else raw.get("min_speakers")
    hi = max_speakers if max_speakers is not None else raw.get("max_speakers")

    # A local pipeline folder wins: it needs no HF cache and no token, because
    # config.yaml resolves its $model/ sub-models relative to its own directory.
    local = model_path if model_path is not None else raw.get("model_path")
    ref = None
    if local and str(local).strip():
        candidate = _expand(local)
        if candidate.is_dir() and (candidate / "config.yaml").exists():
            candidate = candidate / "config.yaml"
        if candidate.exists():
            ref = str(candidate)

    return DiarizationConfig(
        model_id=ref or model_id or raw.get("model_id") or DEFAULT_DIARIZATION_MODEL_ID,
        step=float(step if step is not None else raw.get("step", DEFAULT_DIARIZATION_STEP)),
        token=str(token) if token else None,
        speakers=int(spk) if spk else None,
        min_speakers=int(lo) if lo else None,
        max_speakers=int(hi) if hi else None,
    )
