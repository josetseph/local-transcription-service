"""MLX Whisper transcription engine (Apple Silicon GPU)."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from local_transcription_service.config import WhisperConfig, resolve_model_ref
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
                    "words": [asdict(w) for w in s.words],
                }
                for s in self.segments
            ],
        }


def transcribe_audio(
    audio_path: Path,
    cfg: WhisperConfig,
    *,
    word_timestamps: bool = False,
    show_progress: bool = False,
) -> TranscriptResult:
    """Transcribe one audio file on the Apple Silicon GPU via MLX.

    ``word_timestamps`` adds a per-segment DTW alignment pass; it is only worth
    paying for when the caller actually needs word timings (the json writer).
    ``show_progress`` renders mlx-whisper's own progress bar on stderr.
    """
    # Route HuggingFace downloads to the shared models root. Must be set before
    # huggingface_hub is imported (it reads the cache path at import time), and
    # mlx_whisper pulls it in, so this stays ahead of the import below.
    if cfg.models_root:
        cfg.models_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_CACHE", str(cfg.models_root / "huggingface"))

    import mlx_whisper

    try:
        raw = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=resolve_model_ref(cfg),
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
