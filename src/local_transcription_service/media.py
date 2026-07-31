"""ffmpeg discovery and media normalization for Whisper."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

AUDIO_EXTENSIONS = frozenset(
    {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".opus", ".aiff", ".aif"}
)
VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".wmv", ".mpeg", ".mpg", ".flv"}
)
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def find_ffmpeg() -> str:
    for candidate in ("ffmpeg", "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        path = Path(candidate)
        if path.is_file():
            return str(path)
        found = shutil.which(candidate)
        if found:
            return found
    raise RuntimeError(
        "ffmpeg not found. Install it and ensure it is on PATH "
        "(macOS: brew install ffmpeg; Linux: apt/dnf install ffmpeg; "
        "Windows: https://ffmpeg.org/download.html)."
    )


def is_media_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS


def collect_media_files(path: Path, *, recursive: bool = False) -> list[Path]:
    path = path.resolve()
    if path.is_file():
        if not is_media_file(path):
            raise ValueError(f"Unsupported media type: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")

    pattern = "**/*" if recursive else "*"
    files = sorted(
        p for p in path.glob(pattern) if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No media files found in {path}")
    return files


def extract_whisper_wav(media_path: Path, output_wav: Path | None = None) -> Path:
    """Extract/convert media to 16 kHz mono WAV for Whisper."""
    ffmpeg = find_ffmpeg()
    media_path = media_path.resolve()
    if output_wav is None:
        tmp = tempfile.NamedTemporaryFile(prefix="whisper-", suffix=".wav", delete=False)
        output_wav = Path(tmp.name)
        tmp.close()
    else:
        output_wav = output_wav.resolve()
        output_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(media_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"ffmpeg failed for {media_path}:\n{err}")
    return output_wav
