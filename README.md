# local-transcription-service

Local CLI to transcribe **video and audio**, with speaker labels. Runs on the **Apple Silicon GPU** via [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), or on **CPU/CUDA anywhere else** via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — the right engine is picked for you. One command, configurable model directory (SSD, external drive, or NAS), and no cloud round-trip.

Measured on an M3: **~3.1x realtime** with `whisper-large-v3` — an 81-minute recording transcribes in about 26 minutes while using ~6% CPU. For comparison, the same model under faster-whisper on CPU ran at 0.71x, about 96 minutes.

**Repo:** [github.com/josetseph/local-transcription-service](https://github.com/josetseph/local-transcription-service)

## Prerequisites

- Python **3.11+**
- An **Apple Silicon Mac** for the GPU engine; any platform works on CPU/CUDA
- [ffmpeg](https://ffmpeg.org/) on your `PATH` — `brew install ffmpeg`

## Install

```bash
git clone https://github.com/josetseph/local-transcription-service.git
cd local-transcription-service
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

This installs the `transcribe` command into that environment.

### Use from any terminal (optional)

Put the venv’s `transcribe` on your `PATH` (for example via `~/bin`) and keep a user config at `~/.config/local-transcription-service/config.yaml` so model paths work outside the project directory. Then from any folder:

```bash
transcribe /path/to/file.mp4
# writes meeting.txt / meeting.srt into the current directory
transcribe /path/to/file.mp4 -o ~/Desktop/transcripts
```

## Quick start

```bash
transcribe meeting.mp4
transcribe interview.wav
transcribe ./recordings/
```

By default each input gets a `.txt` and `.srt` in the **current working directory** (where you ran the command). Use `-o` to write somewhere else.

```bash
transcribe talk.mov --language en --format txt,srt,json
transcribe /path/to/clip.mp4 -o ./out/
transcribe ./media --recursive
```

## Configure the Whisper model directory

Point the tool at models on a **local disk**, **external drive**, or **NAS** so you (and others) are not locked to one machine path.

### Option A — environment variables

```bash
# Cache / download root (created if missing)
export WHISPER_MODELS_ROOT="/Volumes/NAS/Projects/models"

# Or use an existing MLX model folder (config.json + weights.safetensors)
export WHISPER_MODEL_PATH="/Volumes/NAS/Projects/models/whisper-large-v3-mlx"

transcribe clip.mp4
```

### Option B — config file

Copy the example and edit paths:

```bash
cp config.example.yaml config.yaml
```

Or use a user-level config:

`~/.config/local-transcription-service/config.yaml`

```yaml
whisper:
  models_root: ~/.cache/whisper-models   # or /path/on/external/drive
  model_path: ""                         # e.g. .../whisper-large-v3-mlx
  model_id: mlx-community/whisper-large-v3-mlx
  language: null
```

### Option C — CLI flags

```bash
transcribe clip.mp4 \
  --models-root /Volumes/NAS/Projects/models \
  --model-path /Volumes/NAS/Projects/models/whisper-large-v3-mlx
```

**Precedence (highest wins):** CLI flags → environment → project `config.yaml` → `~/.config/local-transcription-service/config.yaml` → built-in defaults.

### Local model vs download

- If `model_path` is a directory holding **`config.json`** beside **`weights.safetensors`** (or `weights.npz`), that MLX bundle is used.
- Otherwise the tool loads `model_id` (default `mlx-community/whisper-large-v3-mlx`), downloading into `<models_root>/huggingface`.

Do **not** point `model_path` at a CTranslate2 (`model.bin`) or Hugging Face Transformers checkout — MLX needs its own converted layout. Ready-made bundles live under the [`mlx-community`](https://huggingface.co/mlx-community) org; convert your own with `mlx_whisper.convert`.

First run may download several GB depending on the model.

## Engines

Two backends, selected automatically:

| engine | runs on | platforms |
|--------|---------|-----------|
| `mlx` | Apple Silicon **GPU** | macOS on M1 and later |
| `faster-whisper` | **CPU or CUDA** | macOS, Linux, Windows |

`pip install` pulls the one that fits your machine, so a plain install works
everywhere. `--engine auto` (the default) resolves in this order:

1. an explicit `--engine`, always honoured — so a missing backend gives a precise install error instead of a silent substitution
2. the format of your local `model_path`: `config.json` + weights means MLX, `model.bin` means CTranslate2 — but only if that engine is installed
3. the platform: MLX on Apple Silicon, faster-whisper elsewhere

The two use **different model formats and different model ids**, and they are not
interchangeable. Leave `model_id` unset and the default follows the engine
(`mlx-community/whisper-large-v3-mlx` or `Systran/faster-whisper-large-v3`). If a
configured `model_path` or `model_id` does not match the selected engine, the
tool says so on stderr rather than silently downloading gigabytes of weights it
cannot read.

`--device` and `--compute-type` apply to **faster-whisper only**; MLX always uses
the Apple GPU.

```bash
transcribe clip.mp4                                   # auto
transcribe clip.mp4 --engine faster-whisper --device cuda --compute-type float16
transcribe clip.mp4 --engine mlx                      # force the GPU path
```

Model choice is the remaining speed/accuracy dial:

| `model_id` | Notes |
|--------|--------|
| `mlx-community/whisper-large-v3-mlx` | Default. Full 32-layer decoder |
| `mlx-community/whisper-large-v3-turbo` | 2.6x faster (8.1x realtime), but see the warning below |
| `mlx-community/whisper-small-mlx` | Much faster and lighter; noticeably weaker on hard audio |

### A note on `turbo`

`large-v3-turbo` keeps the 32-layer encoder but cuts the decoder from 32 layers
to 4. On clean read speech the published WER barely moves, which makes it look
like a free 2.6x. On hard audio it is not free: the shallower decoder is a much
weaker language model, and in low-signal stretches it invents text rather than
staying quiet.

Measured over two 4-minute windows of a distant-mic meeting recording — crosstalk,
background noise, long pauses:

| model | repetition loops | drift into other scripts |
|-------|------------------|--------------------------|
| `large-v3-turbo` | yes | yes |
| `large-v3` | no | no |

Turbo also emitted ~20% more words than `large-v3` on the same audio: filler
invented in the dead air, e.g. `"yeah yeah yeah yeah yeah"` and
`"...sitzen kaum Pulgger кли смотреть willkommen"` in an English-only recording.
It is a reasonable pick for a fast rough pass over clean audio, and a poor one
for a record you intend to quote from.

## CLI reference

```text
transcribe PATH [OPTIONS]

  PATH                 Media file or directory

  -o, --output-dir     Write transcripts here (default: current directory)
  -f, --format         txt,srt,vtt,json (default: txt,srt)
  -l, --language       Language code (e.g. en); default auto-detect
  -r, --recursive      Recurse when PATH is a directory
  --models-root        Model download/cache directory
  --model-path         Local model directory (MLX or CTranslate2)
  --model-id           HuggingFace model id (engine-appropriate)
  --engine             auto | mlx | faster-whisper
  --device             faster-whisper only: auto | cpu | cuda
  --compute-type       faster-whisper only: int8, float16, int8_float16
  --word-timestamps    Per-word timings (default: on only for json output)
  --diarize            Label speakers (who spoke when)
  --diarization-step   Seconds between diarization windows (default 2.0)
  --speakers           Exact speaker count, when known
```

Supported media includes common video (`mp4`, `mov`, `mkv`, `webm`, …) and audio (`wav`, `mp3`, `m4a`, `ogg`, `flac`, …). Non-WAV inputs are converted with ffmpeg to 16 kHz mono WAV temporarily, then removed.

## Speaker diarization

Whisper transcribes words; it has no idea *who* is speaking. `--diarize` adds a
second pass ([pyannote.audio](https://github.com/pyannote/pyannote-audio)) that
labels speakers, then attributes each transcribed word to whichever speaker was
talking at that moment.

```bash
pip install -e ".[diarize]"
```

The model is gated: accept the conditions on
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1),
create a read token, then either `export HF_TOKEN=...` or set `diarization.token`
in your config. Weights cache under `<models_root>/huggingface` like everything else.

```bash
transcribe meeting.m4a --diarize
transcribe meeting.m4a --diarize --speakers 4      # when you know the headcount
```

Output gains speaker labels: `SPEAKER_01: …` lines in txt, speaker-prefixed cues
in srt/vtt, and a `speaker` field per segment in json. Diarization attributes
individual words, so `--word-timestamps` turns on automatically.

### Speed, and the `--diarization-step` trade

Diarization runs on **CPU** — pyannote is a PyTorch model with no working Metal
path, so unlike the Whisper pass it cannot use the GPU. It is the slower half of
the pipeline. Roughly 95% of its time goes to one speaker-embedding pass per
analysis window, so cost scales with the number of windows — which is what
`--diarization-step` controls.

Measured on an M3 over a 10-minute meeting recording:

| step | speed | speakers found | agreement with pyannote's default |
|------|-------|----------------|-----------------------------------|
| 1.0 (pyannote default) | 1.83x realtime | 4 | — |
| **2.0 (our default)** | **3.42x realtime** | 4 | **95.7%** |
| 3.0 | 5.17x realtime | **3** — merges two people | 48.1% |

2.0 nearly halves the runtime while changing the result less than switching
pyannote models does. 3.0 is a cliff, not a further trade. Drop to `1.0` if you
would rather have pyannote's stock behaviour.

For an 81-minute recording that works out to roughly 26 minutes of transcription
plus 24 minutes of diarization.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `WHISPER_MODELS_ROOT` | Download/cache root |
| `WHISPER_MODEL_PATH` | Local model folder (MLX or CTranslate2) |
| `WHISPER_MODEL_ID` | HuggingFace model id fallback |
| `WHISPER_ENGINE` | `auto`, `mlx`, or `faster-whisper` |
| `WHISPER_DEVICE` | faster-whisper only: `auto`, `cpu`, `cuda` |
| `WHISPER_COMPUTE_TYPE` | faster-whisper only: `int8`, `float16` |
| `WHISPER_LANGUAGE` | Language code, or empty/`auto` for detect |
| `HF_TOKEN` | HuggingFace token for the gated diarization model |

## Troubleshooting

**`ffmpeg not found`** — Install ffmpeg and confirm `ffmpeg -version` works in the same shell.

**Wrong model folder** — MLX needs `config.json` next to `weights.safetensors` (or `weights.npz`); faster-whisper needs `model.bin`. A mismatch is reported as a warning naming the format it found, and the tool falls back to the engine's default model id.

**`No module named 'mlx_whisper'`** — you asked for the MLX engine on a machine without it. Either install it (Apple Silicon only) or use `--engine faster-whisper`.

**Slow or out-of-memory** — Try `mlx-community/whisper-large-v3-turbo` (2.6x faster, but read the turbo warning above) or a smaller `model_id` (e.g. `mlx-community/whisper-small-mlx`), and drop `--word-timestamps` if you do not need per-word timings — it adds an alignment pass.

**Large first download** — Set `WHISPER_MODELS_ROOT` to a drive with enough free space before the first run.

## License

MIT — see [LICENSE](LICENSE).
