import queue

import numpy as np
import pytest

from local_transcription_service.capture import CaptureUnavailable, Mixer
from local_transcription_service.live import _calibrate


def test_mixer_sums_clips_and_bounds_lag():
    out = []
    mixer = Mixer(out.append)
    mixer.add_mic(np.full(4, 0.1, np.float32))            # no system audio yet
    for level in range(1, 40):                           # 39 frames; only the last 25 are kept
        mixer.add_system(np.full(4, level / 100, np.float32))
    mixer.add_mic(np.full(4, 0.1, np.float32))
    mixer.add_mic(np.full(4, 0.9, np.float32))

    np.testing.assert_allclose(out[0], 0.1)
    np.testing.assert_allclose(out[1], 0.1 + 0.15, rtol=1e-6)
    np.testing.assert_allclose(out[2], 1.0)              # 0.9 + 0.16, clipped
    assert len(mixer._system) == 23


def _silent_queue():
    q = queue.Queue()
    for _ in range(30):
        q.put(np.zeros(320, np.float32))
    return q


def test_calibration_rejects_a_silent_mic_and_returns_every_frame_it_read():
    with pytest.raises(CaptureUnavailable):
        _calibrate(_silent_queue())

    q = queue.Queue()
    for _ in range(70):
        q.put(np.full(320, 0.01, np.float32))
    levels, frames = _calibrate(q)
    assert len(levels) == 60 and len(frames) == 60       # replayed, so none go unrecorded
