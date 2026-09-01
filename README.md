# local-transcription-service

Local CLI to transcribe **video and audio** with [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) on the **Apple Silicon GPU**. One command, configurable model directory (SSD, external drive, or NAS), and no cloud round-trip.

Measured on an M3: **~7.7x realtime** with `whisper-large-v3-turbo` — an 81-minute recording transcribes in about 10 minutes while using ~6% CPU.

**Repo:** [github.com/josetseph/local-transcription-service](https://github.com/josetseph/local-transcription-service)

## Prerequisites

- An **Apple Silicon Mac** (M1 or later) — MLX has no CPU or CUDA fallback
- Python **3.11+**
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
export WHISPER_MODEL_PATH="/Volumes/NAS/Projects/models/whisper-large-v3-turbo-mlx"

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
  model_path: ""                         # e.g. .../whisper-large-v3-turbo-mlx
  model_id: mlx-community/whisper-large-v3-turbo
  language: null
```

### Option C — CLI flags

```bash
transcribe clip.mp4 \
  --models-root /Volumes/NAS/Projects/models \
  --model-path /Volumes/NAS/Projects/models/whisper-large-v3-turbo-mlx
```

**Precedence (highest wins):** CLI flags → environment → project `config.yaml` → `~/.config/local-transcription-service/config.yaml` → built-in defaults.

### Local model vs download

- If `model_path` is a directory holding **`config.json`** beside **`weights.safetensors`** (or `weights.npz`), that MLX bundle is used.
- Otherwise the tool loads `model_id` (default `mlx-community/whisper-large-v3-turbo`), downloading into `<models_root>/huggingface`.

Do **not** point `model_path` at a CTranslate2 (`model.bin`) or Hugging Face Transformers checkout — MLX needs its own converted layout. Ready-made bundles live under the [`mlx-community`](https://huggingface.co/mlx-community) org; convert your own with `mlx_whisper.convert`.

First run may download several GB depending on the model.

## Platform

MLX runs on the GPU of **Apple Silicon Macs only** (M1 and later). There is no CPU, CUDA, or Intel-Mac fallback — the `--device` and `--compute-type` flags from the previous faster-whisper backend are gone, because MLX has nothing to choose between.

Model choice is the remaining speed/accuracy dial:

| `model_id` | Notes |
|--------|--------|
| `mlx-community/whisper-large-v3-turbo` | Default. 4 decoder layers; fastest of the large family |
| `mlx-community/whisper-large-v3-mlx` | Full 32-layer decoder; slower, marginally more accurate |
| `mlx-community/whisper-small-mlx` | Much faster and lighter; noticeably weaker on hard audio |

## CLI reference

```text
transcribe PATH [OPTIONS]

  PATH                 Media file or directory

  -o, --output-dir     Write transcripts here (default: current directory)
  -f, --format         txt,srt,vtt,json (default: txt,srt)
  -l, --language       Language code (e.g. en); default auto-detect
  -r, --recursive      Recurse when PATH is a directory
  --models-root        Model download/cache directory
  --model-path         Local MLX model directory (config.json + weights)
  --model-id           HuggingFace MLX model id
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

For an 81-minute recording that works out to roughly 10 minutes of transcription
plus 24 minutes of diarization.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `WHISPER_MODELS_ROOT` | Download/cache root |
| `WHISPER_MODEL_PATH` | Local MLX folder (`config.json` + weights) |
| `WHISPER_MODEL_ID` | HuggingFace MLX model id fallback |
| `WHISPER_LANGUAGE` | Language code, or empty/`auto` for detect |
| `HF_TOKEN` | HuggingFace token for the gated diarization model |

## Troubleshooting

**`ffmpeg not found`** — Install ffmpeg and confirm `ffmpeg -version` works in the same shell.

**Wrong model folder** — MLX needs `config.json` next to `weights.safetensors`. A missing `config.json` is the usual cause of a load failure; CT2 (`model.bin`) and HF Transformers checkouts are different formats and will not load.

**Slow or out-of-memory** — Try a smaller `model_id` (e.g. `mlx-community/whisper-small-mlx`), and drop `--word-timestamps` if you do not need per-word timings — it adds an alignment pass.

**Large first download** — Set `WHISPER_MODELS_ROOT` to a drive with enough free space before the first run.

## License

MIT — see [LICENSE](LICENSE).
