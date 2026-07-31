"""faster-whisper transcription engine."""

from __future__ import annotations

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


def transcribe_audio(audio_path: Path, cfg: WhisperConfig) -> TranscriptResult:
    from faster_whisper import WhisperModel

    model_ref = resolve_model_ref(cfg)
    model_kwargs: dict = {}
    if cfg.models_root:
        cfg.models_root.mkdir(parents=True, exist_ok=True)
        model_kwargs["download_root"] = str(cfg.models_root)

    model = None
    try:
        model = WhisperModel(
            model_ref,
            device=cfg.device,
            compute_type=cfg.compute_type,
            **model_kwargs,
        )
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=cfg.language,
            word_timestamps=True,
        )
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
