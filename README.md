# local-transcription-service

Local CLI to transcribe **video and audio** — from files, the microphone, or whatever your computer is playing (the other side of a call) — with speaker labels. Runs on the **Apple Silicon GPU** via [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) (default) or [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), or on **CPU/CUDA anywhere else** via [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

Measured on an M3 with the default `qwen3-asr-1.7b`: **~4.7x realtime**, so an 83-minute recording transcribes in roughly 18 minutes. For comparison, faster-whisper on CPU took 96 minutes for an 81-minute recording.

**Repo:** [github.com/josetseph/local-transcription-service](https://github.com/josetseph/local-transcription-service)

## Prerequisites

- Python **3.11+**
- An **Apple Silicon Mac** for the GPU engine; any platform works on CPU/CUDA
- [ffmpeg](https://ffmpeg.org/) on your `PATH` — `brew install ffmpeg`
- macOS 14.4+ to capture system audio on a Mac (`--source system`)

## Install

```bash
git clone https://github.com/josetseph/local-transcription-service.git
cd local-transcription-service
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

This installs the `transcribe` command into that environment.

### First run: choosing a model

The first time you transcribe with no model configured, `transcribe` downloads one.
In a terminal it asks which, and where to put it:

```
Choose a speech model (figures measured on an Apple M3):

  1) Qwen3-ASR 1.7B  (4.7 GB + 1.8 GB aligner)  recommended
     lecture WER 28%, 4.7x realtime · dictation 0% WER, 0.7s per sentence · names: 3 of 6 right, where 0.6B got none
  2) Qwen3-ASR 0.6B  (1.9 GB + 1.8 GB aligner)
     lecture WER 33-35%, 13.9x realtime · dictation 0% WER, 0.4s per sentence
  3) Whisper large-v3  (3.1 GB)
     lecture WER 36%, 2.0x realtime · dictation 8% WER, 1.4s per sentence

Model [1]:
Download to [~/.cache/transcribe-models]:
```

Without a terminal to ask in (a script, a pipe), it takes the recommended model. The
speech model, the aligner that gives Qwen its word timings, and the 30 MB speaker
model land as plain folders, and their paths are saved to
`~/.config/local-transcription-service/config.yaml`, so later runs work offline. Run
`transcribe --setup` to choose again. Off Apple Silicon the choices are faster-whisper
large-v3 and small, which have not been measured. None of the models needs a
HuggingFace account.

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
  language: en        # or auto to detect
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
| `qwen` | Apple Silicon **GPU** | macOS on M1 and later |
| `mlx` | Apple Silicon **GPU** | macOS on M1 and later |
| `faster-whisper` | **CPU or CUDA** | macOS, Linux, Windows |

### Which model

Measured on an M3:

**`qwen3-asr-1.7b` for everything.** It is the default and needs no flags.

| task | measured |
|------|----------|
| recordings — meetings, lectures, interviews | 28.1% WER vs whisper's 36.1% on distant-mic audio |
| live dictation | 0.0% WER at 0.71s per sentence |
| clean, close-mic batch | 0.0% WER |

`qwen3-asr-0.6b` is roughly twice as fast (0.36s per sentence, also 0.0% WER on
clean dictation) but weaker on proper nouns — 0/6 against 1.7B's 3/6 on names in
a meeting recording. One model that handles names is worth a third of a second
per sentence, so 1.7B is the pick everywhere. Reach for 0.6B only if latency
becomes the binding constraint.

Whisper stays available as `--engine mlx` but wins none of these. It is trained
on 30-second windows, so it degrades on the short utterances dictation produces
— in the live test it mangled two of six while both Qwen models were perfect.

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
  -f, --format         txt,srt,vtt,json,md (default: txt,srt)
  -l, --language       Language code (e.g. en), or auto to detect; default en
  -r, --recursive      Recurse when PATH is a directory
  --models-root        Model download/cache directory
  --model-path         Local model directory (MLX or CTranslate2)
  --model-id           HuggingFace model id (engine-appropriate)
  --engine             auto | mlx | faster-whisper
  --device             faster-whisper only: auto | cpu | cuda
  --compute-type       faster-whisper only: int8, float16, int8_float16
  --word-timestamps    Per-word timings (default: on only for json output)
  --diarize            Label speakers (who spoke when)
  --diarize-only       Label a saved live session's transcript, no re-transcription
  --summarize          Head the md file with a title and summary from a local GGUF model
  --record             Record until Ctrl+C, then transcribe the whole recording
  --source SOURCE      What --live/--record hear: mic (default), system, or both
  --setup              Choose and download a speech model, then exit
  --diarization-step   Seconds between diarization windows (default 2.0)
  --speakers           Exact speaker count, when known
  --min-speakers       Lower bound when the count is unknown
  --max-speakers       Upper bound when the count is unknown
  --live               Transcribe the microphone instead of a file
  --input-device       Input device for --live (default: system input)
  --list-devices       List input devices and exit
  --silence            Pause that ends a sentence, seconds (default 0.5)
```

Supported media includes common video (`mp4`, `mov`, `mkv`, `webm`, …) and audio (`wav`, `mp3`, `m4a`, `ogg`, `flac`, …). Non-WAV inputs are converted with ffmpeg to 16 kHz mono WAV temporarily, then removed.

## Live transcription

```bash
transcribe --live                       # prints each sentence as you pause
transcribe --live -l fr                 # another language (default: en)
transcribe --live -o ~/Documents        # write the session elsewhere
transcribe --live --silence 0.4         # cut sooner after you stop talking
transcribe --list-devices               # show input devices
transcribe --live --source both         # a call: you and the other side
```

`PATH` is omitted with `--live`. It uses the system input device; `--input-device`
overrides that.

Ctrl+C stops the session and **always saves it** — to `-o` if given, otherwise the
directory you ran from, named `live-YYYYmmdd-HHMMSS`:

```
live-20260913-101500.wav    the full recording, pauses included
live-20260913-101500.txt    the transcript
live-20260913-101500.srt    timestamped cues — times match the wav exactly
live-20260913-101500.json   word timings — lets you add speakers later
```

The recording streams to disk as it is captured, so a crash keeps everything
already heard.

### Record now, transcribe after

```bash
transcribe --record                           # the microphone
transcribe --record --source both --diarize   # a call, with speaker labels
transcribe --record -o ~/Documents/calls
```

`--record` captures to `record-YYYYmmdd-HHMMSS.wav` — in `-o`, or the directory you
ran it from — without transcribing while it runs. Ctrl+C stops it, and the whole
recording is then transcribed like any file, with the transcript written beside the
wav, along with a `.json` of word timings so `--diarize-only` can add speakers later
without transcribing again. The model hears full context instead of one sentence at a time, and nothing
depends on pauses. `--source`, `--diarize`, `--format` and the rest apply as they do to
files. The model is checked before recording starts, so a long recording never ends
in a setup error.

### Meetings: capturing what the computer plays

The microphone hears you, not the other side of a call — that goes to your speakers.
`--source system` captures what the computer plays; `--source both` mixes it with the
microphone, so both sides land in one transcript.

```bash
transcribe --live --source both          # a call: you and them
transcribe --live --source system        # a webinar or video: only what plays
transcribe --live --source both --diarize
```

Nothing is rerouted and no virtual device is installed; the input and output devices
stay as the system set them.

| Platform | How system audio is captured |
|----------|------------------------------|
| macOS 14.4+ | A Core Audio process tap, read through a private device that only this process can see and that is removed on exit |
| Windows | WASAPI loopback of the default output device |
| Linux | The PulseAudio/PipeWire monitor of the default output device |

The tap copies what every app plays, so it follows whichever output the system has
selected. On an M3, speech played through the built-in speakers and through a Bluetooth
headset came back sample-for-sample (correlation 1.00) — the headset in music mode
(48 kHz) and in call mode with its microphone open (24 kHz) — and the system's input and
output devices were identical before and after every run. Switching the output in the
middle of a session has not been tested. Because it hears every app, notification sounds
and anything else playing end up in the transcript too. No permission prompt appeared on that machine; if a session prints
`no system audio yet` while something is playing, allow your terminal to record system
audio under System Settings > Privacy & Security. **Windows and Linux** go through the
[soundcard](https://github.com/bastibe/SoundCard) library's loopback support and have
not been tested on real hardware yet.


With a call on speakers, the microphone hears the other side as well, so `--source both`
gets them twice, slightly delayed. On a 30-second lecture clip played through the
speakers, `both` disagreed with file mode on 62-73% of words, against 42-47% for
`system` alone. Headphones keep the speakers out of the microphone.

### Speakers later, without transcribing again

```bash
transcribe --diarize-only live-20260913-101500.wav
```

Diarization needs only the audio and the transcript's timestamps, and a `--live` or
`--record` session keeps both. `--diarize-only` diarizes the `.wav`, labels each word in the existing
`.json`, and rewrites that session's `.txt`/`.srt`/`.json` in place — speech
recognition never runs again. It is the same result `--live --diarize` gives, later.

To transcribe the recording again instead — with full context, catching words the live
threshold missed — use `transcribe live-….wav --diarize -o elsewhere/`. Without `-o`
that replaces the live transcript.

### Speakers in a live session

```bash
transcribe --live -l en --diarize
```

Transcription stays live. When you stop, the recording is diarized and each word
is assigned to whoever was speaking at that moment — there is no second
transcription pass, and a sentence is split where the speaker changes inside it.
The plain transcript is written first, so interrupting diarization loses nothing.

Word timings cost about 0.02s per sentence and use the forced-aligner model, set
with `whisper.aligner_path`.

The language defaults to English. Avoid `-l auto` for live work: it detects per
utterance, and a two-second utterance gives the detector very little to go on.

### How sentences are detected

A sentence ends after `--silence` seconds (default 0.5) below a speech threshold.
The threshold follows the room: three times the 10th-percentile loudness of the
last five seconds, refreshed every half second. It is not calibrated once on
whatever is playing when you start — that mattered: calibrating on the median of
the first 1.2s of a lecture that began mid-sentence set the bar above a quiet
student and dropped 63% of their words.

Measured on a five-minute distant-mic lecture and two close-mic dictations:

| | quiet speaker's words | loud speaker's words | close-mic sentences found |
|---|---|---|---|
| previous: median calibration, 0.6s silence | 37% | 92% | 4 |
| rolling floor, 0.5s silence | 85% | 98% | 4 |

Live mode only transcribes what crosses the threshold. For an in-room recording
with speakers at different distances, `transcribe recording.wav --diarize`
transcribes every word rather than 85% of the quietest person's.

## Speaker diarization

Whisper transcribes words; it has no idea *who* is speaking. `--diarize` adds a
second pass ([pyannote.audio](https://github.com/pyannote/pyannote-audio)) that
labels speakers, then attributes each transcribed word to whichever speaker was
talking at that moment.

```bash
pip install -e ".[diarize]"
```

The default model,
[pyannote-community/speaker-diarization-community-1](https://huggingface.co/pyannote-community/speaker-diarization-community-1),
is byte-identical to pyannote's gated community-1 pipeline but ungated, so it needs no
HuggingFace account or token. First-run setup downloads it (30 MB) into a plain folder
and points `diarization.model_path` at it; that folder's `config.yaml` resolves the
sub-models relative to itself, so it also works offline.

```bash
transcribe meeting.m4a --diarize                             # count auto-detected
transcribe meeting.m4a --diarize --speakers 4                # exact headcount
transcribe meeting.m4a --diarize --min-speakers 8 --max-speakers 12
```

The speaker count is **detected automatically** — `--speakers` is optional and
constrains clustering only when you already know the answer. When you have a
rough idea but not an exact number, `--min-speakers` / `--max-speakers` bound the
search, which is safer than guessing an exact count: an exact `--speakers` that
is wrong forces the wrong number of clusters, splitting one person in two or
merging two people into one.

Output gains speaker labels: `SPEAKER_01: …` lines in txt, speaker-prefixed cues
in srt/vtt, and a `speaker` field per segment in json. Diarization attributes
individual words, so `--word-timestamps` turns on automatically.

### Speed, and the `--diarization-step` trade

Diarization runs on the **GPU** where there is one: Metal on Apple Silicon, CUDA
elsewhere, CPU as the fallback. Measured on an M3 (torch 2.13, pyannote 4.0.7)
over a 10-minute lecture slice at the default step: 40s on the GPU against 284s on
CPU, about 15x realtime, with identical speaker turns. Roughly 95% of its time
goes to one speaker-embedding pass per analysis window, so cost scales with the
number of windows — which is what `--diarization-step` controls.

The step comparison below was measured on CPU; the speeds are relative, the
agreement figures do not depend on the device.

Measured on an M3 over a 10-minute meeting recording:

| step | speed | speakers found | agreement with pyannote's default |
|------|-------|----------------|-----------------------------------|
| 1.0 (pyannote default) | 1.83x realtime | 4 | — |
| **2.0 (our default)** | **3.42x realtime** | 4 | **95.7%** |
| 3.0 | 5.17x realtime | **3** — merges two people | 48.1% |

2.0 nearly halves the runtime while changing the result less than switching
pyannote models does. 3.0 is a cliff, not a further trade. Drop to `1.0` if you
would rather have pyannote's stock behaviour.

For an 86-minute recording that works out to roughly 17 minutes of transcription
plus 6 minutes of diarization.

## Notes: markdown and a summary

`-f md` writes the transcript as timestamped lines, one or two sentences each, with
speakers numbered in the order they first speak:

```text
## Transcript
[00:10] Speaker 1: So today is going to be essentially for corrections.
[00:54] Speaker 2: Yeah.
```

`--summarize` puts a title and summary above it: what the session was, three to five
topic sections, next steps with who took them on, and decisions made. It implies
`-f md`. The summary comes from a GGUF chat model run in-process by llama.cpp, after
speech recognition and diarization have released their memory; nothing leaves the
machine and no server is involved.

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install 'llama-cpp-python>=0.3.34'   # the "summarize" extra
transcribe lecture.m4a --diarize --summarize
transcribe --diarize-only record-20260917-132001.wav --summarize     # a saved session
```

```yaml
summary:
  model_path: ~/models/gguf/google_gemma-4-E4B-it-Q4_K_M.gguf
  n_ctx: 32768
```

Measured on an M3 with Gemma 4 E4B (Q4_K_M): an 86-minute class is 15,700 tokens of
summary input, so a 32k window holds about two and a half hours in one pass. Longer
recordings are noted part by part and summarized from the notes. Diarizing and
summarizing that class from its saved transcript took under 9 minutes together.

The summary is only as good as the transcript. Speech recognition mishears names
and jargon, and the model is told to drop what makes no sense rather than guess,
so check figures and names against the transcript below it. If the model fails to
load or the prompt does not fit, the md is still written, without a summary.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `WHISPER_MODELS_ROOT` | Download/cache root |
| `WHISPER_MODEL_PATH` | Local model folder (MLX or CTranslate2) |
| `WHISPER_MODEL_ID` | HuggingFace model id fallback |
| `WHISPER_ENGINE` | `auto`, `mlx`, or `faster-whisper` |
| `WHISPER_DEVICE` | faster-whisper only: `auto`, `cpu`, `cuda` |
| `WHISPER_COMPUTE_TYPE` | faster-whisper only: `int8`, `float16` |
| `WHISPER_LANGUAGE` | Language code, or `auto` to detect (default `en`) |
| `HF_TOKEN` | Only for a gated model you configure yourself; the defaults need none |

## Troubleshooting

**`ffmpeg not found`** — Install ffmpeg and confirm `ffmpeg -version` works in the same shell.

**Wrong model folder** — MLX needs `config.json` next to `weights.safetensors` (or `weights.npz`); faster-whisper needs `model.bin`. A mismatch is reported as a warning naming the format it found, and the tool falls back to the engine's default model id.

**`No module named 'mlx_whisper'`** — you asked for the MLX engine on a machine without it. Either install it (Apple Silicon only) or use `--engine faster-whisper`.

**Slow or out-of-memory** — Try `mlx-community/whisper-large-v3-turbo` (2.6x faster, but read the turbo warning above) or a smaller `model_id` (e.g. `mlx-community/whisper-small-mlx`), and drop `--word-timestamps` if you do not need per-word timings — it adds an alignment pass.

**Large first download** — The recommended model is 6.5 GB with its aligner. Give a folder on a drive with room at the first-run prompt, or set `WHISPER_MODELS_ROOT` before a run with no terminal. Setup checks free space before downloading.

## License

MIT — see [LICENSE](LICENSE).
