# Measurement tools

Scratch tools used to choose the engines. Not part of the package.

## compare.py — batch accuracy and throughput

Records once (or takes `--wav`), transcribes with every engine, scores each
against what you said.

```bash
python tools/compare.py --secs 30 --say "the exact script you will read"
python tools/compare.py --wav rec.wav --say "..." --context "names and jargon"
```

## livemic.py — live dictation latency

Measures what dictation actually costs: the wait between finishing a sentence
and seeing text. Captures on a callback thread and endpoints on silence, then
runs each engine on the utterance.

```bash
python tools/livemic.py --device 3 --say "your script"   # compares all three
python tools/livemic.py --list-devices
```

`--device` matters: a virtual audio device (Sonoroid, Benzene, Loopback) taking
the system default records digital silence, and every engine then scores ~100%
WER for reasons that have nothing to do with the models.

## Results (M3, September 2026)

| model | live dictation | batch, distant mic | batch, clean |
|-------|----------------|--------------------|--------------|
| **qwen3-asr-1.7b** | 0.0% WER, 0.71s | **28.1% WER** | 0.0% WER |
| qwen3-asr-0.6b | 0.0% WER, **0.36s** | 32.6% WER | 3.3% WER |
| whisper-large-v3 | 8.2% WER, 1.42s | 36.1% WER | 0.0% WER |

1.7B is the settled choice for every task: it is never worse than the
alternatives and is the only one that handles proper nouns (3/6 on names where
0.6B managed 0/6). 0.6B buys ~0.35s per sentence if latency ever matters more
than names.

Whisper degrades on short utterances — it is trained on 30s windows, so
sentence-length dictation chunks starve it of context. It mangled two of six
utterances in the live test while both Qwen models were perfect.
