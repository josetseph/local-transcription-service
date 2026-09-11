#!/usr/bin/env python3
"""Record once, transcribe with every engine, score them against what you said.

  ./compare.py --secs 30 --say "the exact script you are about to read"
  ./compare.py --wav some.wav --say "..."      # reuse a recording

Recording once and replaying it through each model is the point: speaking the
script separately per model makes the takes differ more than the models do.
"""
from __future__ import annotations

import argparse, os, re, subprocess, sys, tempfile, time
from pathlib import Path

MODELS = Path.home() / "Projects/models"
os.environ.setdefault("HF_HUB_CACHE", str(MODELS / "huggingface"))
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"

NUMBERS = {"0":"zero","1":"one","2":"two","3":"three","4":"four","5":"five","6":"six",
           "7":"seven","8":"eight","9":"nine","10":"ten","11":"eleven","12":"twelve",
           "13":"thirteen","14":"fourteen","15":"fifteen","20":"twenty","30":"thirty"}


def normalise(text: str) -> list[str]:
    text = re.sub(r"\b\d+\b", lambda m: NUMBERS.get(m.group(0), m.group(0)), text.lower())
    return re.sub(r"[^\w\s']", " ", text).split()


def wer(ref: list[str], hyp: list[str]) -> float:
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1): d[i][0] = i
    for j in range(len(hyp) + 1): d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1,
                          d[i-1][j-1] + (ref[i-1] != hyp[j-1]))
    return d[len(ref)][len(hyp)] / max(1, len(ref))


def record(seconds: int, path: str) -> None:
    print(f"Recording {seconds}s — read your script now...", file=sys.stderr)
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "avfoundation",
         "-i", ":default", "-t", str(seconds), "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", "-y", path])
    if r.returncode != 0:
        sys.exit("Mic capture failed — grant Terminal microphone access in "
                 "System Settings > Privacy & Security > Microphone.")
    print("done.\n", file=sys.stderr)


def run_qwen(wav: str, model: str, context: str) -> str:
    import mlx_qwen3_asr
    r = mlx_qwen3_asr.transcribe(wav, model=str(MODELS / model),
                                 language="English", context=context)
    return r.text or ""


def run_whisper(wav: str, model: str, _context: str) -> str:
    import mlx_whisper
    r = mlx_whisper.transcribe(wav, path_or_hf_repo=str(MODELS / model),
                               language="en", verbose=None)
    return (r.get("text") or "").strip()


ENGINES = [
    ("qwen 0.6B",  run_qwen,    "qwen3-asr-0.6b"),
    ("qwen 1.7B",  run_qwen,    "qwen3-asr-1.7b"),
    ("whisper v3", run_whisper, "whisper-large-v3-mlx"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=int, default=30)
    ap.add_argument("--say", default=None, help="what you will read, for WER")
    ap.add_argument("--wav", default=None)
    ap.add_argument("--context", default="", help="domain terms (qwen only)")
    args = ap.parse_args()

    wav = args.wav
    if not wav:
        wav = tempfile.mktemp(suffix=".wav")
        record(args.secs, wav)

    import soundfile as sf
    dur = len(sf.read(wav, dtype="float32", always_2d=True)[0]) / 16000
    ref = normalise(args.say) if args.say else None
    rows = []

    for label, fn, model in ENGINES:
        t = time.perf_counter()
        try:
            text = fn(wav, model, args.context)
        except Exception as exc:                        # keep going if one engine fails
            print(f"{label}: FAILED — {exc}\n", file=sys.stderr)
            continue
        el = time.perf_counter() - t
        score = wer(ref, normalise(text)) if ref else None
        rows.append((label, el, dur / el, score, text))
        print(f"{BOLD}{label:<12}{RESET} {el:5.1f}s  {dur/el:5.1f}x realtime"
              + (f"  {BOLD}WER {score*100:5.1f}%{RESET}" if score is not None else ""))
        print(f"   {text}\n")

    if ref and rows:
        best = min(rows, key=lambda r: r[3])
        print(f"{BOLD}best accuracy:{RESET} {best[0]} at {best[3]*100:.1f}% WER "
              f"({best[2]:.1f}x realtime)")
        fastest = max(rows, key=lambda r: r[2])
        if fastest[0] != best[0]:
            print(f"{DIM}fastest: {fastest[0]} at {fastest[2]:.1f}x "
                  f"({fastest[3]*100:.1f}% WER){RESET}")
    print(f"\n{DIM}audio: {wav}{RESET}")


if __name__ == "__main__":
    main()
