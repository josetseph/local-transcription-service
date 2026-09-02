"""Speaker diarization via pyannote.audio, and merging it onto transcript words.

Measured on an M3 over a 10-minute meeting chunk (see README): the pipeline's
default 1.0s segmentation step runs at 1.83x realtime, while a 2.0s step runs at
3.42x and still agrees with the default on 95.7% of speech — a smaller change
than the gap between two different pyannote pipelines. 3.0s collapses two
speakers into one, so DEFAULT_STEP stops at 2.0.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from local_transcription_service.memory import release_accelerator_memory

DEFAULT_MODEL_ID = "pyannote/speaker-diarization-community-1"
DEFAULT_STEP = 2.0


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class DiarizationConfig:
    model_id: str
    step: float
    token: str | None
    speakers: int | None
    min_speakers: int | None = None
    max_speakers: int | None = None


def diarize_audio(
    audio_path: Path,
    cfg: DiarizationConfig,
    models_root: Path | None = None,
    *,
    show_progress: bool = False,
) -> list[Turn]:
    """Return speaker turns for one audio file, sorted by start time.

    ``show_progress`` renders pyannote's own per-stage progress bars. Without
    them a long diarization pass is entirely silent, which is indistinguishable
    from a hang — it is the slower half of the pipeline.
    """
    # Same cache the Whisper side uses; must precede the huggingface_hub import.
    if models_root is not None:
        os.environ.setdefault("HF_HUB_CACHE", str(models_root / "huggingface"))

    import warnings

    import torch
    from pyannote.audio import Pipeline

    # Emitted per chunk from pooling when a speaker is active for a single
    # frame; harmless, but it buries the progress bar in repeated noise.
    warnings.filterwarnings(
        "ignore",
        message=r"std\(\): degrees of freedom is <= 0",
        category=UserWarning,
    )

    pipeline = None
    try:
        pipeline = Pipeline.from_pretrained(cfg.model_id, token=cfg.token)
        if pipeline is None:
            raise RuntimeError(
                f"Could not load {cfg.model_id}. Gated models need a HuggingFace token "
                "(set HF_TOKEN) and acceptance of the model's conditions."
            )
        pipeline.to(torch.device("cpu"))

        # Fewer, wider windows are the one lever that actually speeds this up:
        # cost scales with the number of windows, since ~95% of the time is the
        # per-window speaker-embedding pass.
        if cfg.step:
            pipeline._segmentation.step = cfg.step

        # An exact count pins clustering; a min/max pair bounds the search when
        # the count is unknown. Unconstrained, clustering decides on its own.
        kwargs: dict = {}
        if cfg.speakers:
            kwargs["num_speakers"] = cfg.speakers
        else:
            if cfg.min_speakers:
                kwargs["min_speakers"] = cfg.min_speakers
            if cfg.max_speakers:
                kwargs["max_speakers"] = cfg.max_speakers
        if show_progress:
            from pyannote.audio.pipelines.utils.hook import ProgressHook

            with ProgressHook() as hook:
                output = pipeline(str(audio_path), hook=hook, **kwargs)
        else:
            output = pipeline(str(audio_path), **kwargs)

        # pyannote 4.x returns DiarizeOutput; earlier versions a bare Annotation.
        annotation = getattr(output, "speaker_diarization", output)
        turns = [
            Turn(start=float(seg.start), end=float(seg.end), speaker=str(spk))
            for seg, _, spk in annotation.itertracks(yield_label=True)
        ]
        turns.sort(key=lambda t: t.start)
        return turns
    finally:
        if pipeline is not None:
            del pipeline
        release_accelerator_memory()


def speaker_at(turns: list[Turn], when: float) -> str | None:
    """Speaker active at ``when``; falls back to the nearest turn.

    Whisper words routinely land in diarization gaps (breaths, short pauses),
    so an unlabelled word is worse than one attributed to its closest turn.
    """
    if not turns:
        return None
    for turn in turns:
        if turn.start <= when <= turn.end:
            return turn.speaker
    nearest = min(turns, key=lambda t: t.start - when if when < t.start else when - t.end)
    return nearest.speaker


def apply_diarization(result, turns: list[Turn]):
    """Split each transcript segment at speaker changes.

    Words are attributed by midpoint. Splitting *within* Whisper's existing
    segments rather than regrouping across them keeps cues subtitle-length —
    grouping purely by speaker produced single cues over 30s long. The txt
    writer merges consecutive same-speaker segments back for readability.
    """
    from local_transcription_service.whisper_engine import (
        Segment,
        TranscriptResult,
        join_words,
    )

    if not turns:
        return result

    segments: list[Segment] = []
    for seg in result.segments:
        if not seg.words:
            # No word timings to attribute: label the whole segment by midpoint.
            segments.append(
                Segment(
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    words=[],
                    speaker=speaker_at(turns, (seg.start + seg.end) / 2),
                )
            )
            continue

        current: list = []
        current_speaker: str | None = None

        def flush() -> None:
            if current:
                segments.append(
                    Segment(
                        text=join_words(current),
                        start=current[0].start,
                        end=current[-1].end,
                        words=list(current),
                        speaker=current_speaker,
                    )
                )

        for word in seg.words:
            speaker = speaker_at(turns, (word.start + word.end) / 2)
            if speaker != current_speaker:
                flush()
                current, current_speaker = [], speaker
            current.append(word)
        flush()

    return TranscriptResult(
        text=result.text,
        language=result.language,
        segments=segments,
    )
