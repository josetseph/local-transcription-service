# local-transcription-service

Local CLI to transcribe **video and audio** with [faster-whisper](https://github.com/SYSTRAN/faster-whisper). One command, configurable model directory (SSD, external drive, or NAS), and CPU / CUDA / Apple Silicon–friendly defaults.

**Repo:** [github.com/josetseph/local-transcription-service](https://github.com/josetseph/local-transcription-service)

## Prerequisites

- Python **3.11+**
- [ffmpeg](https://ffmpeg.org/) on your `PATH`
  - macOS: `brew install ffmpeg`
  - Linux: install via your package manager (`ffmpeg`)
  - Windows: [ffmpeg builds](https://ffmpeg.org/download.html) and add to `PATH`

## Install

```bash
git clone https://github.com/josetseph/local-transcription-service.git
cd local-transcription-service
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

This installs the `transcribe` command.

## Quick start

```bash
transcribe meeting.mp4
transcribe interview.wav
transcribe ./recordings/
```

By default each input gets a `.txt` and `.srt` next to the source file.

```bash
transcribe talk.mov --language en --format txt,srt,json -o ./out/
transcribe ./media --recursive
```

## Configure the Whisper model directory

Point the tool at models on a **local disk**, **external drive**, or **NAS** so you (and others) are not locked to one machine path.

### Option A — environment variables

```bash
# Cache / download root (created if missing)
export WHISPER_MODELS_ROOT="/Volumes/NAS/Projects/models"

# Or use an existing CTranslate2 folder (must contain model.bin)
export WHISPER_MODEL_PATH="/Volumes/NAS/Projects/models/faster-whisper-large-v3"

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
  model_path: ""                         # e.g. .../faster-whisper-large-v3
  model_id: Systran/faster-whisper-large-v3
  device: auto
  compute_type: int8
  language: null
```

### Option C — CLI flags

```bash
transcribe clip.mp4 \
  --models-root /Volumes/NAS/Projects/models \
  --model-path /Volumes/NAS/Projects/models/faster-whisper-large-v3
```

**Precedence (highest wins):** CLI flags → environment → project `config.yaml` → `~/.config/local-transcription-service/config.yaml` → built-in defaults.

### Local CT2 vs download

- If `model_path` is a directory that contains **`model.bin`**, that CTranslate2 bundle is used.
- Otherwise the tool loads `model_id` (default `Systran/faster-whisper-large-v3`) and stores downloads under `models_root`.

Do **not** point `model_path` at a Hugging Face safetensors / Transformers Whisper folder. faster-whisper needs a **CTranslate2** layout (`model.bin`).

First run may download several GB depending on the model.

## Devices and platforms

| Setting | Notes |
|--------|--------|
| `device: auto` | Lets CTranslate2 pick **CPU** or **CUDA** |
| `compute_type: int8` | Default; good memory footprint on most machines |
| `float16` / `int8_float16` | Often faster on NVIDIA GPUs with enough VRAM |
| Apple Silicon | Typically runs on **CPU** via CTranslate2 (not MPS) |

Override with env (`WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`) or flags (`--device`, `--compute-type`).

Works on **macOS**, **Linux**, and **Windows** as long as Python and ffmpeg are available.

## CLI reference

```text
transcribe PATH [OPTIONS]

  PATH                 Media file or directory

  -o, --output-dir     Write transcripts here (default: beside each source)
  -f, --format         txt,srt,vtt,json (default: txt,srt)
  -l, --language       Language code (e.g. en); default auto-detect
  -r, --recursive      Recurse when PATH is a directory
  --models-root        Model download/cache directory
  --model-path         Local CT2 model directory (model.bin)
  --model-id           HuggingFace faster-whisper id
  --device             auto | cpu | cuda
  --compute-type       e.g. int8, float16, int8_float16
```

Supported media includes common video (`mp4`, `mov`, `mkv`, `webm`, …) and audio (`wav`, `mp3`, `m4a`, `ogg`, `flac`, …). Non-WAV inputs are converted with ffmpeg to 16 kHz mono WAV temporarily, then removed.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `WHISPER_MODELS_ROOT` | Download/cache root |
| `WHISPER_MODEL_PATH` | Local CT2 folder with `model.bin` |
| `WHISPER_MODEL_ID` | HuggingFace model id fallback |
| `WHISPER_DEVICE` | `auto`, `cpu`, `cuda` |
| `WHISPER_COMPUTE_TYPE` | e.g. `int8`, `float16` |
| `WHISPER_LANGUAGE` | Language code, or empty/`auto` for detect |

## Troubleshooting

**`ffmpeg not found`** — Install ffmpeg and confirm `ffmpeg -version` works in the same shell.

**Wrong model folder** — If you see load errors, check that `model_path` contains `model.bin` (CT2). HF Transformers Whisper checkouts are a different format.

**Slow or out-of-memory** — Try a smaller `model_id` (e.g. `Systran/faster-whisper-small`), keep `compute_type: int8`, or use `--device cpu`.

**Large first download** — Set `WHISPER_MODELS_ROOT` to a drive with enough free space before the first run.

## License

MIT — see [LICENSE](LICENSE).
