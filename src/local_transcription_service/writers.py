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


def format_srt(segments: list[Segment]) -> str:
    blocks: list[str] = []
    for i, seg in enumerate(segments, start=1):
        text = seg.text.strip()
        if not text:
            continue
        blocks.append(f"{i}\n{_ts_srt(seg.start)} --> {_ts_srt(seg.end)}\n{text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def format_vtt(segments: list[Segment]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        text = seg.text.strip()
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
        path.write_text(result.text + ("\n" if result.text else ""), encoding="utf-8")
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
