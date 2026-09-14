"""First-run model setup: pick a speech model, download it, remember the choice.

Models land as plain folders under the chosen directory rather than the
HuggingFace cache, and their paths go into the user config, so later runs work
offline. Every repo below is ungated: no HuggingFace account or token.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import yaml

from local_transcription_service.config import (
    USER_CONFIG,
    WhisperConfig,
    _deep_merge,
    _load_yaml,
    default_model_id,
    is_apple_silicon,
)

# What mlx-qwen3-asr itself fetches from a Qwen repo; the rest is documentation.
QWEN_FILES = ["*.json", "*.safetensors", "*.txt", "*.model"]


@dataclass(frozen=True)
class Model:
    folder: str
    label: str
    repo: str
    gb: float
    notes: str = ""
    aligner: bool = False          # needs the forced aligner for word timings
    allow: list[str] | None = None
    ignore: list[str] | None = None


# Figures from the engine trials on an Apple M3: WER on a distant-mic lecture
# recording, speed as a multiple of realtime, and the wait per sentence live.
APPLE_SILICON = (
    Model("qwen3-asr-1.7b", "Qwen3-ASR 1.7B", "Qwen/Qwen3-ASR-1.7B", 4.7,
          "lecture WER 28%, 4.7x realtime · dictation 0% WER, 0.7s per sentence · "
          "names: 3 of 6 right, where 0.6B got none", aligner=True, allow=QWEN_FILES),
    Model("qwen3-asr-0.6b", "Qwen3-ASR 0.6B", "Qwen/Qwen3-ASR-0.6B", 1.9,
          "lecture WER 33-35%, 13.9x realtime · dictation 0% WER, 0.4s per sentence",
          aligner=True, allow=QWEN_FILES),
    Model("whisper-large-v3-mlx", "Whisper large-v3", "mlx-community/whisper-large-v3-mlx", 3.1,
          "lecture WER 36%, 2.0x realtime · dictation 8% WER, 1.4s per sentence"),
)
OTHER = (
    Model("faster-whisper-large-v3", "Whisper large-v3", "Systran/faster-whisper-large-v3", 3.1,
          "not measured on this kind of machine"),
    Model("faster-whisper-small", "Whisper small", "Systran/faster-whisper-small", 0.5,
          "not measured on this kind of machine"),
)
ALIGNER = Model("qwen3-forced-aligner-0.6b", "Qwen3 forced aligner", "Qwen/Qwen3-ForcedAligner-0.6B",
                1.8, "word timings for --live, --diarize and json", allow=QWEN_FILES)
# Byte-identical to the gated pyannote/speaker-diarization-community-1.
DIARIZER = Model("pyannote-community-1", "pyannote community-1",
                 "pyannote-community/speaker-diarization-community-1", 0.03, "speaker labels",
                 ignore=["*.gif"])


def needs_setup(cfg: WhisperConfig, engine: str) -> bool:
    """True when no model is configured and the engine's default is not cached.

    A configured path or id is the user's own choice and is left alone. The cache
    check keeps installs that already downloaded the default by id from being
    asked again.
    """
    if cfg.model_path is not None or cfg.model_id:
        return False
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    # Where the engines download to by id: HF_HUB_CACHE when set, otherwise
    # models_root/huggingface (qwen, mlx) or models_root itself (faster-whisper).
    caches = {os.environ.get("HF_HUB_CACHE"), str(cfg.models_root / "huggingface"), str(cfg.models_root)}
    for cache in caches - {None}:
        try:
            snapshot_download(default_model_id(engine), cache_dir=cache, local_files_only=True)
            return False
        except LocalEntryNotFoundError:
            continue
    return True


def run_setup(cfg: WhisperConfig) -> Path:
    """Download a speech model — asking which, in a terminal — and save its path.

    Without a terminal to ask in, the recommended model is downloaded. Returns
    the chosen model's folder.
    """
    choices = APPLE_SILICON if is_apple_silicon() else OTHER
    if sys.stdin.isatty():
        measured = " (figures measured on an Apple M3)" if choices is APPLE_SILICON else ""
        click.echo(f"\nChoose a speech model{measured}:\n")
        for number, model in enumerate(choices, 1):
            size = f"{model.gb:.1f} GB" + (f" + {ALIGNER.gb:.1f} GB aligner" if model.aligner else "")
            click.echo(f"  {number}) {model.label}  ({size}){'  recommended' if number == 1 else ''}")
            click.echo(f"     {model.notes}")
        pick = choices[click.prompt("\nModel", type=click.IntRange(1, len(choices)), default=1) - 1]
        root = Path(click.prompt("Download to", default=str(cfg.models_root))).expanduser().resolve()
    else:
        pick, root = choices[0], cfg.models_root
        click.echo(
            f"No speech model configured: downloading the recommended one, {pick.label}, to {root}. "
            "Run `transcribe --setup` in a terminal to choose another.",
            err=True,
        )

    parts = [pick, *([ALIGNER] if pick.aligner else []), DIARIZER]
    root.mkdir(parents=True, exist_ok=True)
    # Credit what an interrupted earlier run already fetched, so a resume that
    # fits is not refused.
    have = sum(f.stat().st_size for p in parts for f in (root / p.folder).rglob("*") if f.is_file())
    need = sum(p.gb for p in parts) - have / 1e9
    free = shutil.disk_usage(root).free / 1e9
    if need > free:
        raise click.ClickException(f"{root} has {free:.1f} GB free; the download needs {need:.1f} GB more.")

    from huggingface_hub import snapshot_download

    for part in parts:
        target = root / part.folder
        click.echo(f"Downloading {part.label} ({part.gb:.2g} GB) to {target}")
        snapshot_download(part.repo, local_dir=target, allow_patterns=part.allow,
                          ignore_patterns=part.ignore)

    whisper = {"models_root": str(root), "model_path": str(root / pick.folder)}
    if pick.aligner:
        whisper["aligner_path"] = str(root / ALIGNER.folder)
    path = USER_CONFIG.expanduser()
    saved = _deep_merge(_load_yaml(path), {
        "whisper": whisper,
        "diarization": {"model_path": str(root / DIARIZER.folder)},
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(saved, sort_keys=False), encoding="utf-8")
    click.echo(f"Saved to {path}\n")
    return root / pick.folder
