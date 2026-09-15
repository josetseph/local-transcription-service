"""Audio sources for --live and --record: the microphone, what the computer plays, or both.

System audio is the other side of a meeting. It never reaches the microphone —
it goes to the speakers — so it is captured separately, without rerouting
anything: no virtual device, no change to the system input or output.

- macOS 14.4+: a Core Audio process tap copies the output mix into a private
  aggregate device that only this process can see, destroyed on exit. On an M3,
  a chirp played through the speakers came back at 0.985 correlation, with the
  system devices identical before and after.
- Windows: WASAPI loopback of the default output, via soundcard.
- Linux: the PulseAudio/PipeWire monitor of the default output, via soundcard.
  Neither of these two has been run on real hardware yet.
"""

from __future__ import annotations

import collections
import contextlib
import os
import sys
import threading
import time
import uuid

SOURCES = ("mic", "system", "both")


class CaptureUnavailable(RuntimeError):
    """No usable capture source."""


@contextlib.contextmanager
def open_source(source, on_frame, *, rate, frame, device=None):
    """Call ``on_frame`` with mono float32 frames of ``frame`` samples until exit."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; choose from {', '.join(SOURCES)}")
    with contextlib.ExitStack() as stack:
        if source == "both":
            mixer = Mixer(on_frame)
            # System first: on macOS it re-scans PortAudio's devices, which would
            # close a microphone stream that was already open.
            stack.enter_context(_system(mixer.add_system, rate, frame))
            stack.enter_context(_mic(mixer.add_mic, rate, frame, device))
        elif source == "system":
            stack.enter_context(_system(on_frame, rate, frame))
        else:
            stack.enter_context(_mic(on_frame, rate, frame, device))
        yield


class Mixer:
    """Sum microphone and system audio into one stream, clocked by the microphone.

    ponytail: two devices never share a clock. Drift is absorbed by dropping
    system frames once MAX_LAG are queued, or by mixing in nothing when none are,
    rather than by resampling — an occasional 20ms glitch. Resample if a long
    session ever sounds broken.
    """

    MAX_LAG = 25  # system frames (0.5s) held while the microphone catches up

    def __init__(self, on_frame):
        self._emit = on_frame
        # Appended by the system thread, popped only by the mic thread: deque
        # operations are atomic, so no lock.
        self._system = collections.deque(maxlen=self.MAX_LAG)

    def add_system(self, chunk):
        self._system.append(chunk)

    def add_mic(self, chunk):
        import numpy as np

        mixed = chunk.copy()
        if self._system:
            other = self._system.popleft()[: len(mixed)]
            mixed[: len(other)] += other
        self._emit(np.clip(mixed, -1.0, 1.0))


@contextlib.contextmanager
def _mic(on_frame, rate, frame, device):
    """Read an input device on its own thread, through PortAudio's blocking API.

    Samples wait in PortAudio's buffer, which fills without Python. The callback
    API instead runs Python on the audio thread, and with the speech model
    transcribing in the same process that thread was starved: 2.2 s of a
    35-second capture never arrived, and no overflow was reported. A blocking
    read with a 1-second buffer lost nothing under the same load, where the
    smaller "high" preset overflowed 8 times.
    """
    import sounddevice as sd

    stop = threading.Event()
    try:
        stream = sd.InputStream(samplerate=rate, channels=1, dtype="float32",
                                blocksize=frame, device=device, latency=1.0)
    except (ValueError, sd.PortAudioError) as exc:   # e.g. an --input-device that does not exist
        raise CaptureUnavailable(f"could not open the input device: {exc}") from exc

    def run():
        while not stop.is_set():
            # Poll rather than block: a macOS tap delivers nothing while idle,
            # and a read blocked on it would never let the thread stop.
            if stream.read_available < frame:
                time.sleep(0.005)
                continue
            data, _overflowed = stream.read(frame)
            on_frame(data[:, 0].copy())

    with stream:
        worker = threading.Thread(target=run, name="audio-capture", daemon=True)
        worker.start()
        try:
            yield
        finally:
            stop.set()
            worker.join(timeout=2)


@contextlib.contextmanager
def _system(on_frame, rate, frame):
    if sys.platform == "darwin":
        with _macos_tap() as device, _fill_gaps(on_frame, rate, frame) as emit, \
                _mic(emit, rate, frame, device):
            yield
    else:
        with _loopback(on_frame, rate, frame):
            yield


@contextlib.contextmanager
def _fill_gaps(on_frame, rate, frame, gap=1.0):
    """Pass frames through, adding silence while the source delivers none.

    A macOS process tap sends no frames at all while nothing is playing
    (measured: 0 per second idle, 50 once audio starts), which would squeeze
    every silence out of the session timeline, and the transcript's timestamps
    with it. Silence is added only once the frames delivered fall a full ``gap``
    behind the clock: a callback that is merely late on a busy machine then
    catches up with its own frames instead of arriving after inserted silence.
    ponytail: paced by the wall clock, not the device clock; ppm drift between
    the two would add ``gap`` of silence every few hours.
    """
    import numpy as np

    seconds = frame / rate
    lock, stop = threading.Lock(), threading.Event()
    began, emitted = time.monotonic(), [0]

    def emit(chunk):
        with lock:
            emitted[0] += 1
            on_frame(chunk)

    def fill():
        while not stop.wait(0.1):
            with lock:
                behind = int((time.monotonic() - began) / seconds) - emitted[0]
                if behind * seconds >= gap:
                    for _ in range(behind):
                        on_frame(np.zeros(frame, np.float32))
                    emitted[0] += behind

    worker = threading.Thread(target=fill, name="system-audio-gaps", daemon=True)
    worker.start()
    try:
        yield emit
    finally:
        stop.set()
        worker.join(timeout=1)


@contextlib.contextmanager
def _macos_tap():
    """Yield a PortAudio device index that reads the system output mix."""
    import platform

    version = platform.mac_ver()[0] or "0"
    if tuple(int(p) for p in version.split(".")[:2]) < (14, 4):
        raise CaptureUnavailable(f"capturing system audio needs macOS 14.4 or later; this is {version}.")
    try:
        import CoreAudio
    except ImportError as exc:
        raise CaptureUnavailable(
            "capturing system audio needs PyObjC's CoreAudio bindings: "
            "pip install pyobjc-framework-CoreAudio"
        ) from exc

    description = CoreAudio.CATapDescription.alloc().initMonoGlobalTapButExcludeProcesses_([])
    description.setPrivate_(True)
    status, tap = CoreAudio.AudioHardwareCreateProcessTap(description, None)
    if status:
        raise CaptureUnavailable(f"macOS refused to tap system audio (Core Audio status {status}).")
    name = f"transcribe system audio {os.getpid()}"
    try:
        # Plain string keys: PyObjC exposes the kAudioAggregateDevice*Key
        # constants as bytes, and passing those crashes CreateAggregateDevice.
        status, aggregate = CoreAudio.AudioHardwareCreateAggregateDevice({
            "name": name,
            "uid": f"local-transcription-service.{uuid.uuid4()}",
            "private": True,          # visible to this process only
            "stacked": False,
            "tapautostart": True,
            "taps": [{"uid": description.UUID().UUIDString(), "drift": True}],
        }, None)
        if status:
            raise CaptureUnavailable(f"could not open the system-audio tap (Core Audio status {status}).")
        try:
            import sounddevice as sd

            # PortAudio lists devices once, when it initialises; re-scan so the
            # aggregate appears. ponytail: private sounddevice API.
            sd._terminate()
            sd._initialize()
            index = next((i for i, d in enumerate(sd.query_devices()) if d["name"] == name), None)
            if index is None:
                raise CaptureUnavailable("the system-audio tap did not appear as an input device.")
            yield index
        finally:
            CoreAudio.AudioHardwareDestroyAggregateDevice(aggregate)
    finally:
        CoreAudio.AudioHardwareDestroyProcessTap(tap)


@contextlib.contextmanager
def _loopback(on_frame, rate, frame):
    """Windows and Linux: record the default output through soundcard."""
    started, stop, failure = threading.Event(), threading.Event(), []

    def run():
        try:
            # Imported on this thread: on Windows, soundcard initialises COM
            # (CoInitializeEx) in the importing thread, so setup and use share one.
            import soundcard

            speaker = soundcard.default_speaker()
            # Windows lists each output again as a loopback input under the same
            # id; PulseAudio and PipeWire name it "<sink>.monitor".
            source_id = speaker.id if sys.platform == "win32" else f"{speaker.id}.monitor"
            microphone = soundcard.get_microphone(source_id, include_loopback=True)
            with microphone.recorder(samplerate=rate, blocksize=frame) as recorder:
                started.set()
                while not stop.is_set():
                    # All channels, mixed down; silence while nothing plays.
                    on_frame(recorder.record(numframes=frame).mean(axis=1).astype("float32"))
        except Exception as exc:
            failure.append(exc)
            started.set()

    worker = threading.Thread(target=run, name="system-audio", daemon=True)
    worker.start()
    if not started.wait(10) or failure:
        stop.set()
        reason = failure[0] if failure else "no response from the audio system"
        raise CaptureUnavailable(f"could not capture system audio: {reason}")
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=2)
