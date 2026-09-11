#!/usr/bin/env python3
"""Live mic dictation through all three engines at once.

Speak, pause, and each engine prints its transcription of that utterance with
the wait you actually experienced. One shared mic capture, so the three see
identical audio.

  ./livemic.py
  ./livemic.py --say "your script"     # prints per-engine WER at the end
"""
from __future__ import annotations

import argparse, collections, os, queue, re, signal, sys, time
from pathlib import Path

import numpy as np

MODELS = Path.home() / "Projects/models"
os.environ.setdefault("HF_HUB_CACHE", str(MODELS / "huggingface"))
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
COLORS = ["\033[36m", "\033[33m", "\033[35m"]          # cyan, yellow, magenta
RATE = 16000

NUMBERS = {"0":"zero","1":"one","2":"two","3":"three","4":"four","5":"five","6":"six",
           "7":"seven","8":"eight","9":"nine","10":"ten","13":"thirteen","20":"twenty"}


def normalise(t: str) -> list[str]:
    t = re.sub(r"[,](?=\d)", "", t)
    t = re.sub(r"\b\d+\b", lambda m: NUMBERS.get(m.group(0), m.group(0)), t.lower())
    return re.sub(r"[^\w\s']", " ", t).split()


def wer(ref: list[str], hyp: list[str]) -> float:
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1): d[i][0] = i
    for j in range(len(hyp) + 1): d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1] + (ref[i-1] != hyp[j-1]))
    return d[len(ref)][len(hyp)] / max(1, len(ref))


def qwen(model):
    def go(chunk, ctx):
        import mlx_qwen3_asr
        return (mlx_qwen3_asr.transcribe(chunk, model=str(MODELS / model),
                                         language="English", context=ctx).text or "").strip()
    return go


def whisper(model):
    def go(chunk, _ctx):
        import mlx_whisper
        return (mlx_whisper.transcribe(chunk, path_or_hf_repo=str(MODELS / model),
                                       language="en", verbose=None).get("text") or "").strip()
    return go


ENGINES = [("qwen 0.6B", qwen("qwen3-asr-0.6b")),
           ("qwen 1.7B", qwen("qwen3-asr-1.7b")),
           ("whisper  ", whisper("whisper-large-v3-mlx"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--say", default=None)
    ap.add_argument("--context", default="")
    ap.add_argument("--silence", type=float, default=0.6,
                    help="pause length that ends an utterance (seconds)")
    ap.add_argument("--min-utt", type=float, default=0.5)
    ap.add_argument("--preroll", type=float, default=0.4,
                    help="audio kept from before speech is detected (seconds)")
    ap.add_argument("--sensitivity", type=float, default=2.5,
                    help="speech threshold as a multiple of the noise floor")
    ap.add_argument("--max-utt", type=float, default=15.0,
                    help="force-emit an utterance after this long")
    ap.add_argument("--device", default=None,
                    help="input device name or index (default: system default)")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    import sounddevice as sd

    if args.list_devices:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                mark = " <-- system default" if i == sd.default.device[0] else ""
                print(f"  [{i}] {d['name']}{mark}")
        return

    device = args.device
    if device is not None and device.isdigit():
        device = int(device)
    name = sd.query_devices(device, kind="input")["name"]
    print(f"{DIM}input: {name}{RESET}", file=sys.stderr)

    stop = {"now": False, "printed": False}
    collected: list[list[str]] = [[] for _ in ENGINES]
    lat: list[list[float]] = [[] for _ in ENGINES]
    ref = normalise(args.say) if args.say else None

    def summarise() -> None:
        if stop["printed"]:
            return
        stop["printed"] = True
        n = max((len(x) for x in lat), default=0)
        print(f"\n{BOLD}summary over {n} utterances{RESET}")
        for i, (label, _) in enumerate(ENGINES):
            if not lat[i]:
                continue
            full = " ".join(t for t in collected[i] if t and not t.startswith("<"))
            score = f"   WER {wer(ref, normalise(full))*100:5.1f}%" if ref else ""
            print(f"  {COLORS[i]}{label}{RESET} mean {np.mean(lat[i]):5.2f}s  "
                  f"worst {max(lat[i]):5.2f}s{score}")

    def on_sigint(_s, _f):
        if stop["now"]:
            summarise()                      # never exit without the numbers
            os._exit(130)
        stop["now"] = True
        print(f"\n{DIM}stopping — finishing this utterance...{RESET}", file=sys.stderr)

    signal.signal(signal.SIGINT, on_sigint)

    print("warming up models (once, so it does not count against latency)...",
          file=sys.stderr)
    warm = np.zeros(RATE, dtype=np.float32)
    for _, fn in ENGINES:
        try:
            fn(warm, args.context)
        except Exception as exc:
            print(f"  warmup failed: {exc}", file=sys.stderr)

    collected: list[list[str]] = [[] for _ in ENGINES]
    lat: list[list[float]] = [[] for _ in ENGINES]

    frame = int(0.02 * RATE)
    # Transcription blocks for ~2s per utterance. If capture shares that thread
    # the stream is never drained and everything spoken meanwhile is dropped, so
    # audio goes into a queue from the callback and is consumed here.
    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def on_audio(indata, _frames, _t, status):
        if status:
            print(f"{DIM}[audio: {status}]{RESET}", file=sys.stderr)
        audio_q.put(np.asarray(indata, dtype=np.float32).reshape(-1).copy())

    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                        blocksize=frame, callback=on_audio, device=device):
        print(f"{DIM}calibrating noise floor — stay quiet for 1.5s...{RESET}", file=sys.stderr)
        # The stream's first frames are digital silence while the device spins
        # up. Including them drags the floor to zero, the threshold lands under
        # room tone, and the endpointer then never fires at all.
        amb: list[float] = []
        deadline = time.time() + 4.0
        while len(amb) < 60 and time.time() < deadline:
            try:
                rms = float(np.sqrt((audio_q.get(timeout=0.5) ** 2).mean()))
            except queue.Empty:
                break
            if rms > 1e-6:
                amb.append(rms)
        if len(amb) < 20:
            print(f"{DIM}!! mic delivered {len(amb)} usable frames — is it muted or "
                  f"in use by another app?{RESET}", file=sys.stderr)
        floor = float(np.median(amb)) if amb else 0.002
        thresh = max(floor * args.sensitivity, 1e-4)
        print(f"{DIM}noise floor {floor:.5f}, speech above {thresh:.5f}{RESET}")
        print(f"{BOLD}speak — pause between sentences. Ctrl+C to stop.{RESET}\n")

        buf: list[np.ndarray] = []
        preroll = collections.deque(maxlen=max(1, int(args.preroll / 0.02)))
        silence, speaking, n = 0.0, False, 0
        captured, started = 0.0, time.time()

        while not stop["now"]:
            try:
                chunk = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            rms = float(np.sqrt((chunk ** 2).mean()))
            preroll.append(chunk)

            if rms > thresh:
                if not speaking:
                    speaking, silence = True, 0.0
                    buf = list(preroll)
                buf.append(chunk)
                silence = 0.0
            elif speaking:
                buf.append(chunk)
                silence += 0.02

            # A too-low threshold means silence is never seen; cap the buffer so
            # that degrades into long utterances instead of emitting nothing.
            overlong = speaking and len(buf) * 0.02 >= args.max_utt
            if speaking and (silence >= args.silence or overlong):
                    utt = np.concatenate(buf)
                    speaking, buf, silence = False, [], 0.0
                    if overlong:
                        print(f"{DIM}!! {args.max_utt:.0f}s cap hit — no pause detected; "
                              f"try --sensitivity 4{RESET}", file=sys.stderr)
                    if len(utt) / RATE < args.min_utt:
                        continue
                    n += 1
                    captured += len(utt) / RATE
                    backlog = audio_q.qsize() * 0.02
                    print(f"\033[K{BOLD}#{n}{RESET} {DIM}({len(utt)/RATE:.1f}s speech"
                          f"{f', {backlog:.1f}s queued' if backlog > 0.5 else ''}){RESET}")
                    for i, (label, fn) in enumerate(ENGINES):
                        t = time.perf_counter()
                        try:
                            text = fn(utt, args.context)
                        except Exception as exc:
                            text = f"<failed: {exc}>"
                        el = time.perf_counter() - t
                        lat[i].append(el)
                        collected[i].append(text)
                        print(f"  {COLORS[i]}{label}{RESET} {BOLD}{el:5.2f}s{RESET}  {text}")
                    print()

    if buf:
        utt = np.concatenate(buf)
        if len(utt) / RATE >= args.min_utt:
            print(f"\033[K{BOLD}#{len(lat[0]) + 1}{RESET} {DIM}({len(utt)/RATE:.1f}s "
                  f"speech, final){RESET}")
            for i, (label, fn) in enumerate(ENGINES):
                t = time.perf_counter()
                try:
                    text = fn(utt, args.context)
                except Exception as exc:
                    text = f"<failed: {exc}>"
                lat[i].append(time.perf_counter() - t)
                collected[i].append(text)
                print(f"  {COLORS[i]}{label}{RESET} {text}")
    summarise()


if __name__ == "__main__":
    main()
