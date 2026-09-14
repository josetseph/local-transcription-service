"""Live transcription of the microphone or system audio: endpoint on silence, transcribe each utterance.

Utterances are written to a temporary wav and handed to the normal
``transcribe_audio`` path, so --live works with whichever engine the config
selects rather than only the one this module happened to import.
"""

from __future__ import annotations

import collections
import queue
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from local_transcription_service.capture import CaptureUnavailable, open_source

RATE = 16000
FRAME_SECONDS = 0.02
# The speech threshold is a multiple of a rolling noise floor: the 10th
# percentile of recent frame loudness. A percentile holds up while people are
# talking, where a median rises with speech; rolling means a session that starts
# mid-sentence is not permanently calibrated against speech. Calibrating once on
# the median put the bar above a quiet speaker in a lecture recording, dropping
# 63% of their words.
FLOOR_WINDOW_SECONDS = 5.0
FLOOR_PERCENTILE = 10
FLOOR_REFRESH_FRAMES = 25
MIN_THRESHOLD = 0.0025


@dataclass
class Utterance:
    index: int
    start: float
    end: float
    text: str
    latency: float
    # (word, start, end) on the session timeline, when the transcriber gives them
    words: list = field(default_factory=list)


SILENCE_NOTICE_FRAMES = int(10 / FRAME_SECONDS)


def _calibrate(audio_q) -> tuple[list[float], list]:
    """Measure the room before listening, and hand back every frame read doing it.

    Levels skip the digital silence of device spin-up: including those frames
    drags the floor to zero, which puts the speech threshold below room tone —
    the endpointer then never fires at all. The frames are returned so the caller
    can replay them; dropping them cut the start of every session, so a
    30-second clip came back as a 28.8-second recording.
    """
    import numpy as np

    levels: list[float] = []
    chunks: list = []
    deadline = time.time() + 4.0
    while len(levels) < 60 and time.time() < deadline:
        try:
            chunk = audio_q.get(timeout=0.5)
        except queue.Empty:
            break
        chunks.append(chunk)
        rms = float(np.sqrt((chunk**2).mean()))
        if rms > 1e-6:
            levels.append(rms)
    if len(levels) < 20:
        raise CaptureUnavailable(
            f"the input device delivered only {len(levels)} usable frames of audio. "
            "Check that it is unmuted and that this terminal has microphone "
            "permission. --list-devices shows the alternatives."
        )
    return levels, chunks


def stream_utterances(
    transcribe,
    *,
    device=None,
    silence: float = 0.5,
    min_utterance: float = 0.5,
    max_utterance: float = 30.0,
    preroll: float = 0.4,
    record=None,
    sensitivity: float = 3.0,
    should_stop=lambda: False,
    on_ready=None,
    source: str = "mic",
    on_silence=None,
):
    """Yield an ``Utterance`` each time the speaker pauses.

    ``transcribe`` takes a wav path and returns text, or a result whose segments
    carry word timings relative to that clip. ``record``, if given, is
    called with every captured frame — pauses included — so the caller can keep
    a continuous recording whose timeline matches the utterance timestamps.
    ``source`` is mic, system or both (see capture.py); ``on_silence`` is called
    once if nothing but digital silence has arrived after 10 seconds.
    """
    import numpy as np

    frame = int(FRAME_SECONDS * RATE)
    audio_q: queue.Queue = queue.Queue()

    with open_source(source, audio_q.put, rate=RATE, frame=frame, device=device):
        # System audio has no room tone to measure: it is digital silence until
        # something plays, and waiting for sound would calibrate on speech. It
        # starts at the minimum threshold and lets the rolling floor take over.
        levels, replay = ([], []) if source == "system" else _calibrate(audio_q)
        pending = collections.deque(replay)
        history = collections.deque(maxlen=int(FLOOR_WINDOW_SECONDS / FRAME_SECONDS))

        def floor_threshold(sample) -> float:
            if not sample:                       # nothing but digital silence so far
                return MIN_THRESHOLD
            level = float(np.percentile(np.fromiter(sample, dtype=float), FLOOR_PERCENTILE))
            return max(level * sensitivity, MIN_THRESHOLD)

        threshold = floor_threshold(levels)
        if on_ready is not None:
            on_ready(threshold)

        buf: list = []
        recent = collections.deque(maxlen=max(1, int(preroll / FRAME_SECONDS)))
        quiet, speaking, index = 0.0, False, 0
        clock, frames = 0.0, 0

        while not should_stop():
            if pending:                          # calibration's frames: recorded and heard too
                chunk = pending.popleft()
            else:
                try:
                    chunk = audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue
            clock += FRAME_SECONDS
            if record is not None:
                record(chunk)
            recent.append(chunk)
            rms = float(np.sqrt((chunk**2).mean()))
            # Refresh from frames already heard, then add this one, so the
            # threshold judging a frame never includes that frame.
            if frames and frames % FLOOR_REFRESH_FRAMES == 0:
                threshold = floor_threshold(history)
            if rms > 1e-6:
                history.append(rms)
            frames += 1
            if on_silence is not None and frames == SILENCE_NOTICE_FRAMES and not history:
                on_silence()

            if rms > threshold:
                if not speaking:
                    # Speech is underway before RMS crosses the threshold, so
                    # start from the recent frames or every word onset is
                    # clipped. `recent` already ends with this chunk: appending
                    # it again repeated a 20ms frame in every utterance and
                    # reported each start one frame early.
                    speaking, buf = True, list(recent)
                else:
                    buf.append(chunk)
                quiet = 0.0
            elif speaking:
                buf.append(chunk)
                quiet += FRAME_SECONDS

            if speaking and (quiet >= silence or len(buf) * FRAME_SECONDS >= max_utterance):
                audio = np.concatenate(buf)
                speaking, buf, quiet = False, [], 0.0
                seconds = len(audio) / RATE
                if seconds < min_utterance:
                    continue
                index += 1
                yield _transcribe_chunk(transcribe, audio, index, clock - seconds, seconds)

        if buf:                                    # never discard buffered speech
            audio = np.concatenate(buf)
            if len(audio) / RATE >= min_utterance:
                seconds = len(audio) / RATE
                yield _transcribe_chunk(transcribe, audio, index + 1, clock - seconds, seconds)


def _transcribe_chunk(transcribe, audio, index: int, start: float, seconds: float) -> Utterance:
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        path = Path(handle.name)
    try:
        sf.write(str(path), audio, RATE, subtype="PCM_16")
        began = time.perf_counter()
        result = transcribe(path)
        latency = time.perf_counter() - began
    finally:
        path.unlink(missing_ok=True)
    # Word timings come back relative to this clip, whose first sample sits at
    # ``start`` on the session timeline — shift them there.
    if isinstance(result, str):
        text, words = result, []
    else:
        text = getattr(result, "text", "") or ""
        words = [(w.word, start + w.start, start + w.end)
                 for seg in (getattr(result, "segments", None) or []) for w in seg.words]
    return Utterance(index=index, start=start, end=start + seconds,
                     text=text.strip(), latency=latency, words=words)


def list_input_devices() -> list[tuple[int, str, bool]]:
    import sounddevice as sd

    default = sd.default.device[0]
    return [
        (i, d["name"], i == default)
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]
