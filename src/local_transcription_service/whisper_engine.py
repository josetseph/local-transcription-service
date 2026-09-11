"""Whisper transcription: MLX on Apple Silicon, faster-whisper elsewhere."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from local_transcription_service.config import (
    ENGINE_MLX,
    ENGINE_QWEN,
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
    if engine == ENGINE_QWEN:
        raise EngineUnavailable(
            "The 'qwen' engine needs mlx-qwen3-asr, which runs only on Apple Silicon "
            "(M1 or later).\n"
            "  On an Apple Silicon Mac:  pip install mlx-qwen3-asr\n"
            "  On any other machine:     use --engine faster-whisper"
        )
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

    if engine == ENGINE_QWEN:
        _require("mlx_qwen3_asr", engine)
        return _transcribe_qwen(audio_path, cfg, model_ref, word_timestamps, show_progress)
    if engine == ENGINE_MLX:
        _require("mlx_whisper", engine)
        return _transcribe_mlx(audio_path, cfg, model_ref, word_timestamps, show_progress)
    _require("faster_whisper", engine)
    return _transcribe_faster_whisper(audio_path, cfg, model_ref, word_timestamps)


def _transcribe_qwen(
    audio_path: Path,
    cfg: WhisperConfig,
    model_ref: str,
    word_timestamps: bool,
    show_progress: bool,
) -> TranscriptResult:
    import mlx_qwen3_asr

    try:
        raw = mlx_qwen3_asr.transcribe(
            str(audio_path),
            model=model_ref,
            # Qwen wants a language name ("English"), not an ISO code.
            language=_qwen_language(cfg.language),
            return_timestamps=word_timestamps,
            context=cfg.context or "",
            verbose=show_progress,
        )

        # With return_timestamps the model emits one entry per word, not
        # segments containing words. Regroup into sentence-length cues so srt
        # output is readable, keeping the word timings diarization needs.
        entries = [_as_dict(x) for x in (getattr(raw, "segments", None) or [])]
        words = [
            WordTiming(
                word=str(e.get("text") or "").strip(),
                start=float(e.get("start") or 0.0),
                end=float(e.get("end") or 0.0),
            )
            for e in entries
            if str(e.get("text") or "").strip()
        ]
        # The aligner strips punctuation from each word, but raw text keeps it
        # and the two are token-for-token identical. Restore it, so cues break
        # on sentences and the transcript stays readable.
        tokens = (getattr(raw, "text", "") or "").split()
        if len(tokens) == len(words):
            for word, token in zip(words, tokens):
                word.word = token
        segments = _group_words(words)
        full_text = (getattr(raw, "text", "") or "").strip()
        if not segments and full_text:
            # No timestamps requested: one segment covering the whole file.
            segments = [Segment(text=full_text, start=0.0, end=0.0)]
        detected = getattr(raw, "language", None) or cfg.language
        return TranscriptResult(text=full_text, language=detected, segments=segments)
    finally:
        release_accelerator_memory()


MAX_CUE_SECONDS = 12.0
MAX_CUE_GAP = 1.0


def _group_words(words: list[WordTiming]) -> list[Segment]:
    """Sentence-length cues from word timings: break on . ? ! a pause, or length."""
    segments: list[Segment] = []
    current: list[WordTiming] = []

    def flush() -> None:
        if current:
            segments.append(
                Segment(
                    text=" ".join(w.word for w in current).strip(),
                    start=current[0].start,
                    end=current[-1].end,
                    words=list(current),
                )
            )

    for word in words:
        if current:
            gap = word.start - current[-1].end
            too_long = word.end - current[0].start > MAX_CUE_SECONDS
            if gap > MAX_CUE_GAP or too_long:
                flush()
                current = []
        current.append(word)
        if word.word.endswith((".", "?", "!")):
            flush()
            current = []
    flush()
    return segments


def _as_dict(obj) -> dict:
    """Segments arrive as dicts or dataclass-ish objects depending on version."""
    if isinstance(obj, dict):
        return obj
    return {k: getattr(obj, k) for k in ("text", "start", "end") if hasattr(obj, k)}


# Qwen names languages; everything else here speaks ISO codes.
_QWEN_LANGUAGES = {
    "en": "English", "zh": "Chinese", "fr": "French", "de": "German",
    "es": "Spanish", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "pt": "Portuguese", "ru": "Russian", "ar": "Arabic",
}


def _qwen_language(code: str | None) -> str | None:
    if not code:
        return None
    return _QWEN_LANGUAGES.get(code.lower(), code)


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
