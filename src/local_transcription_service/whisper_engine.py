"""Whisper transcription: MLX on Apple Silicon, faster-whisper elsewhere."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from local_transcription_service.config import (
    ENGINE_MLX,
    WhisperConfig,
    resolve_engine,
    resolve_model_ref,
)
from local_transcription_service.memory import release_accelerator_memory


@dataclass
class WordTiming:
    word: str
    start: float
    end: float


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: list[WordTiming] = field(default_factory=list)
    speaker: str | None = None


@dataclass
class TranscriptResult:
    text: str
    language: str | None
    segments: list[Segment]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "segments": [
                {
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                    "speaker": s.speaker,
                    "words": [asdict(w) for w in s.words],
                }
                for s in self.segments
            ],
        }


def join_words(words: list[WordTiming]) -> str:
    """Rebuild text from stripped word timings (punctuation stays attached)."""
    return " ".join(w.word for w in words if w.word).strip()


class EngineUnavailable(RuntimeError):
    """The selected backend is not installed."""


def _require(module: str, engine: str) -> None:
    from importlib.util import find_spec

    try:
        available = find_spec(module) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        available = False
    if available:
        return
    if engine == ENGINE_MLX:
        raise EngineUnavailable(
            "The 'mlx' engine needs mlx-whisper, which runs only on Apple Silicon "
            "(M1 or later).\n"
            "  On an Apple Silicon Mac:  pip install 'mlx-whisper>=0.4'\n"
            "  On any other machine:     pip install 'faster-whisper>=1.0' "
            "and re-run with --engine faster-whisper"
        )
    raise EngineUnavailable(
        "The 'faster-whisper' engine needs faster-whisper.\n"
        "  pip install 'faster-whisper>=1.0'"
    )


def transcribe_audio(
    audio_path: Path,
    cfg: WhisperConfig,
    *,
    word_timestamps: bool = False,
    show_progress: bool = False,
) -> TranscriptResult:
    """Transcribe one audio file with whichever backend ``cfg`` selects.

    ``word_timestamps`` adds a per-segment alignment pass; it is only worth
    paying for when the caller needs word timings (json output, diarization).
    """
    engine = resolve_engine(cfg)
    model_ref, _ = resolve_model_ref(cfg, engine)

    # Route HuggingFace downloads to the shared models root. Must be set before
    # huggingface_hub is imported (it reads the cache path at import time).
    if cfg.models_root:
        cfg.models_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_CACHE", str(cfg.models_root / "huggingface"))

    if engine == ENGINE_MLX:
        _require("mlx_whisper", engine)
        return _transcribe_mlx(audio_path, cfg, model_ref, word_timestamps, show_progress)
    _require("faster_whisper", engine)
    return _transcribe_faster_whisper(audio_path, cfg, model_ref, word_timestamps)


def _transcribe_mlx(
    audio_path: Path,
    cfg: WhisperConfig,
    model_ref: str,
    word_timestamps: bool,
    show_progress: bool,
) -> TranscriptResult:
    import mlx_whisper

    try:
        raw = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=model_ref,
            language=cfg.language,
            word_timestamps=word_timestamps,
            # mlx-whisper shows its progress bar on verbose=False and hides it
            # on None; True would print every segment instead.
            verbose=False if show_progress else None,
        )

        segments: list[Segment] = []
        for seg in raw.get("segments", []):
            words = [
                WordTiming(word=w["word"].strip(), start=w["start"], end=w["end"])
                for w in (seg.get("words") or [])
                if w.get("word") and w["word"].strip()
            ]
            segments.append(
                Segment(
                    text=seg["text"].strip(),
                    start=seg["start"],
                    end=seg["end"],
                    words=words,
                )
            )
        full_text = (raw.get("text") or "").strip()
        if not full_text:
            full_text = " ".join(s.text for s in segments if s.text).strip()
        detected = raw.get("language") or cfg.language
        return TranscriptResult(text=full_text, language=detected, segments=segments)
    finally:
        release_accelerator_memory()


def _transcribe_faster_whisper(
    audio_path: Path,
    cfg: WhisperConfig,
    model_ref: str,
    word_timestamps: bool,
) -> TranscriptResult:
    from faster_whisper import WhisperModel

    model = None
    try:
        model_kwargs = {"download_root": str(cfg.models_root)} if cfg.models_root else {}
        model = WhisperModel(
            model_ref,
            device=cfg.device,
            compute_type=cfg.compute_type,
            **model_kwargs,
        )
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=cfg.language,
            word_timestamps=word_timestamps,
        )
        # Exhaust the lazy iterator while the model is still alive.
        segments: list[Segment] = []
        for seg in segments_iter:
            words = [
                WordTiming(word=w.word.strip(), start=w.start, end=w.end)
                for w in (seg.words or [])
                if w.word and w.word.strip()
            ]
            segments.append(
                Segment(
                    text=seg.text.strip(),
                    start=seg.start,
                    end=seg.end,
                    words=words,
                )
            )
        full_text = " ".join(s.text for s in segments if s.text).strip()
        detected = getattr(info, "language", None) or cfg.language
        return TranscriptResult(text=full_text, language=detected, segments=segments)
    finally:
        if model is not None:
            del model
        release_accelerator_memory()
