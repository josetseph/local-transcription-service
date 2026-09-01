"""Write transcript outputs (txt, srt, vtt, json)."""

from __future__ import annotations

import json
from pathlib import Path

from local_transcription_service.whisper_engine import Segment, TranscriptResult

SUPPORTED_FORMATS = frozenset({"txt", "srt", "vtt", "json"})


def _ts_srt(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _ts_vtt(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _cue_text(seg: Segment) -> str:
    text = seg.text.strip()
    return f"{seg.speaker}: {text}" if seg.speaker and text else text


def format_txt(segments: list[Segment], fallback: str) -> str:
    """Speaker-labelled lines when diarized, else the plain transcript."""
    if not any(seg.speaker for seg in segments):
        return fallback + ("\n" if fallback else "")
    lines: list[str] = []
    last: str | None = None
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if seg.speaker == last and lines:      # same speaker continuing
            lines[-1] += " " + text
        else:
            lines.append(f"{seg.speaker}: {text}" if seg.speaker else text)
            last = seg.speaker
    return "\n".join(lines) + ("\n" if lines else "")


def format_srt(segments: list[Segment]) -> str:
    blocks: list[str] = []
    for i, seg in enumerate(segments, start=1):
        text = _cue_text(seg)
        if not text:
            continue
        blocks.append(f"{i}\n{_ts_srt(seg.start)} --> {_ts_srt(seg.end)}\n{text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def format_vtt(segments: list[Segment]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        text = _cue_text(seg)
        if not text:
            continue
        lines.append(f"{_ts_vtt(seg.start)} --> {_ts_vtt(seg.end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def write_outputs(
    result: TranscriptResult,
    output_stem: Path,
    formats: set[str],
) -> list[Path]:
    unknown = formats - SUPPORTED_FORMATS
    if unknown:
        raise ValueError(f"Unsupported formats: {', '.join(sorted(unknown))}")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if "txt" in formats:
        path = output_stem.with_suffix(".txt")
        path.write_text(format_txt(result.segments, result.text), encoding="utf-8")
        written.append(path)

    if "srt" in formats:
        path = output_stem.with_suffix(".srt")
        path.write_text(format_srt(result.segments), encoding="utf-8")
        written.append(path)

    if "vtt" in formats:
        path = output_stem.with_suffix(".vtt")
        path.write_text(format_vtt(result.segments), encoding="utf-8")
        written.append(path)

    if "json" in formats:
        path = output_stem.with_suffix(".json")
        path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
        written.append(path)

    return written
