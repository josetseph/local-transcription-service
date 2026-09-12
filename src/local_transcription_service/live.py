"""Live microphone transcription: endpoint on silence, transcribe each utterance.

Utterances are written to a temporary wav and handed to the normal
``transcribe_audio`` path, so --live works with whichever engine the config
selects rather than only the one this module happened to import.
"""

from __future__ import annotations

import collections
import queue
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

RATE = 16000
FRAME_SECONDS = 0.02


@dataclass
class Utterance:
    index: int
    start: float
    end: float
    text: str
    latency: float


class MicUnavailable(RuntimeError):
    """No usable capture device."""


def _calibrate(audio_q, sensitivity: float) -> float:
    """Noise floor, ignoring the digital silence a device emits while spinning up.

    Including those frames drags the floor to zero, which puts the speech
    threshold below room tone — the endpointer then never fires at all.
    """
    import numpy as np

    levels: list[float] = []
    deadline = time.time() + 4.0
    while len(levels) < 60 and time.time() < deadline:
        try:
            chunk = audio_q.get(timeout=0.5)
        except queue.Empty:
            break
        rms = float(np.sqrt((chunk**2).mean()))
        if rms > 1e-6:
            levels.append(rms)
    if len(levels) < 20:
        raise MicUnavailable(
            f"the input device delivered only {len(levels)} usable frames of audio. "
            "Check that it is unmuted and that this terminal has microphone "
            "permission. --list-devices shows the alternatives."
        )
    return max(float(np.median(levels)) * sensitivity, 0.0025)


def stream_utterances(
    transcribe,
    *,
    device=None,
    silence: float = 0.6,
    min_utterance: float = 0.5,
    max_utterance: float = 30.0,
    preroll: float = 0.4,
    keep_audio: list | None = None,
    sensitivity: float = 2.5,
    should_stop=lambda: False,
    on_ready=None,
):
    """Yield an ``Utterance`` each time the speaker pauses.

    ``transcribe`` takes a wav path and returns text.
    """
    import numpy as np
    import sounddevice as sd

    frame = int(FRAME_SECONDS * RATE)
    audio_q: queue.Queue = queue.Queue()

    def on_audio(indata, _frames, _time, _status):
        audio_q.put(np.asarray(indata, dtype=np.float32).reshape(-1).copy())

    with sd.InputStream(
        samplerate=RATE, channels=1, dtype="float32",
        blocksize=frame, callback=on_audio, device=device,
    ):
        threshold = _calibrate(audio_q, sensitivity)
        if on_ready is not None:
            on_ready(threshold)

        buf: list = []
        recent = collections.deque(maxlen=max(1, int(preroll / FRAME_SECONDS)))
        quiet, speaking, index = 0.0, False, 0
        clock = 0.0

        while not should_stop():
            try:
                chunk = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            clock += FRAME_SECONDS
            recent.append(chunk)
            rms = float(np.sqrt((chunk**2).mean()))

            if rms > threshold:
                if not speaking:
                    # Speech is underway before RMS crosses the threshold, so
                    # prepend recent frames or every word onset is clipped.
                    speaking, quiet, buf = True, 0.0, list(recent)
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
                if keep_audio is not None:
                    keep_audio.append(audio)
                yield _transcribe_chunk(transcribe, audio, index, clock - seconds, seconds)

        if buf:                                    # never discard buffered speech
            audio = np.concatenate(buf)
            if len(audio) / RATE >= min_utterance:
                seconds = len(audio) / RATE
                if keep_audio is not None:
                    keep_audio.append(audio)
                yield _transcribe_chunk(transcribe, audio, index + 1, clock - seconds, seconds)


def _transcribe_chunk(transcribe, audio, index: int, start: float, seconds: float) -> Utterance:
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        path = Path(handle.name)
    try:
        sf.write(str(path), audio, RATE, subtype="PCM_16")
        began = time.perf_counter()
        text = transcribe(path)
        latency = time.perf_counter() - began
    finally:
        path.unlink(missing_ok=True)
    return Utterance(index=index, start=start, end=start + seconds,
                     text=text.strip(), latency=latency)


def list_input_devices() -> list[tuple[int, str, bool]]:
    import sounddevice as sd

    default = sd.default.device[0]
    return [
        (i, d["name"], i == default)
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]
