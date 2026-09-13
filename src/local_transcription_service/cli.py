"""Click CLI entry point: ``transcribe``."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click

from local_transcription_service import __version__
from local_transcription_service.config import (
    ENGINE_MLX,
    ENGINES,
    resolve_diarization_config,
    resolve_engine,
    resolve_model_ref,
    resolve_whisper_config,
)
from local_transcription_service.media import collect_media_files, extract_whisper_wav
from local_transcription_service.whisper_engine import transcribe_audio
from local_transcription_service.writers import SUPPORTED_FORMATS, write_outputs


def _parse_formats(value: str) -> set[str]:
    parts = {p.strip().lower() for p in value.split(",") if p.strip()}
    if not parts:
        raise click.BadParameter("provide at least one format")
    unknown = parts - SUPPORTED_FORMATS
    if unknown:
        raise click.BadParameter(
            f"unsupported format(s): {', '.join(sorted(unknown))}; "
            f"choose from {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    return parts


def _run_diarize_only(path, diar_cfg, output_dir, format_set, cfg) -> None:
    """Label speakers on an existing transcript, without transcribing again.

    Diarization needs only audio plus the transcript's timestamps. A live session
    keeps both, so running speech recognition again would redo work already on disk.
    """
    import json

    from local_transcription_service.diarize import apply_diarization, diarize_audio
    from local_transcription_service.whisper_engine import TranscriptResult
    from local_transcription_service.writers import write_outputs

    path = path.resolve()
    stem = path.with_suffix("")
    transcript = stem.with_suffix(".json")
    audio = stem.with_suffix(".wav") if path.suffix.lower() == ".json" else path
    if not transcript.exists():
        raise click.ClickException(
            f"no transcript at {transcript}. --diarize-only reads the .json saved beside "
            "a live recording; to label any other file use --diarize."
        )
    if not audio.exists():
        raise click.ClickException(f"no recording at {audio}.")

    result = TranscriptResult.from_dict(json.loads(transcript.read_text(encoding="utf-8")))
    if not result.segments:
        raise click.ClickException(f"{transcript} has no segments to label.")
    missing = sum(1 for seg in result.segments if not seg.words)
    if missing:
        click.echo(
            f"  note: {missing} of {len(result.segments)} sentences have no word timings; "
            "those are labelled whole, which misses speaker changes inside a sentence.",
            err=True,
        )

    click.echo(f"Recording:  {audio}")
    click.echo(f"Transcript: {transcript} ({len(result.segments)} segments, not re-transcribed)")
    turns = diarize_audio(audio, diar_cfg, models_root=cfg.models_root,
                          show_progress=sys.stderr.isatty())
    result = apply_diarization(result, turns)
    found = sorted({t.speaker for t in turns})
    click.echo(f"  speakers: {len(found)} ({', '.join(found)})")

    target = output_dir.resolve() if output_dir is not None else stem.parent
    target.mkdir(parents=True, exist_ok=True)
    for out in write_outputs(result, target / stem.name, format_set | {"json"}):
        click.echo(f"  wrote: {out}")


def _run_live(cfg, model_ref, engine, want_words, input_device, silence,
              output_dir, format_set, diar_cfg) -> None:
    """Transcribe the microphone until interrupted, then write the session.

    The full recording streams to disk as it is captured, pauses included, so
    the transcript's session timestamps line up with the wav and a crash keeps
    everything already heard. With --diarize, speakers are assigned afterwards
    from that recording, word by word — no second transcription pass.
    """
    import signal

    import soundfile as sf

    from local_transcription_service.live import RATE, MicUnavailable, stream_utterances
    from local_transcription_service.whisper_engine import Segment, TranscriptResult, WordTiming
    from local_transcription_service.writers import write_outputs

    device = int(input_device) if input_device and input_device.isdigit() else input_device
    stopping = {"now": False}

    def on_sigint(_sig, _frame):
        stopping["now"] = True

    signal.signal(signal.SIGINT, on_sigint)

    def transcribe_one(wav_path):
        from local_transcription_service.whisper_engine import transcribe_audio

        # Word timings let --diarize split a sentence where the speaker changes
        # mid-utterance; measured at +0.02s per utterance on an M3. Requested
        # only when diarizing or writing json, matching file mode.
        if want_words:
            return transcribe_audio(wav_path, cfg, word_timestamps=True)
        return transcribe_audio(wav_path, cfg, word_timestamps=False).text

    stem_dir = (output_dir if output_dir is not None else Path.cwd()).resolve()
    stem_dir.mkdir(parents=True, exist_ok=True)
    stem = stem_dir / time.strftime("live-%Y%m%d-%H%M%S")
    wav_out = stem.with_suffix(".wav")

    click.echo(f"Engine:     {engine} ({model_ref})")
    click.echo(f"Recording:  {wav_out}")
    click.echo("Listening.  Pause between sentences. Ctrl+C to stop.\n")

    segments: list[Segment] = []
    try:
        with sf.SoundFile(str(wav_out), "w", samplerate=RATE, channels=1,
                          subtype="PCM_16") as recording:
            for utterance in stream_utterances(
                transcribe_one,
                device=device,
                silence=silence,
                should_stop=lambda: stopping["now"],
                record=recording.write,
            ):
                if not utterance.text:
                    continue
                click.echo(f"  {utterance.text}")
                segments.append(
                    Segment(
                        text=utterance.text,
                        start=utterance.start,
                        end=utterance.end,
                        words=[WordTiming(word=w, start=a, end=b) for w, a, b in utterance.words],
                    )
                )
    except MicUnavailable as exc:
        wav_out.unlink(missing_ok=True)
        raise click.ClickException(str(exc)) from exc

    # Capture is over; let a further Ctrl+C interrupt diarization normally.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    click.echo(f"\n  wrote: {wav_out}")
    if not segments:
        click.echo("Nothing transcribed.")
        return

    text = " ".join(s.text for s in segments)
    result = TranscriptResult(text=text, language=cfg.language, segments=segments)
    click.echo(f"{len(segments)} sentences, {len(text.split())} words")

    # Write the plain transcript first, so interrupting diarization loses nothing.
    written = write_outputs(result, stem, format_set)

    if diar_cfg is not None:
        from local_transcription_service.diarize import apply_diarization, diarize_audio

        click.echo("Diarizing the recording...")
        turns = diarize_audio(wav_out, diar_cfg, models_root=cfg.models_root,
                              show_progress=sys.stderr.isatty())
        result = apply_diarization(result, turns)
        found = sorted({t.speaker for t in turns})
        click.echo(f"  speakers: {len(found)} ({', '.join(found)})")
        written = write_outputs(result, stem, format_set)

    for out in written:
        click.echo(f"  wrote: {out}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("path", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory for transcript files (default: current working directory).",
)
@click.option(
    "-f",
    "--format",
    "formats",
    default="txt,srt",
    show_default=True,
    help="Comma-separated outputs: txt,srt,vtt,json.",
)
@click.option(
    "-l",
    "--language",
    default=None,
    help="Language code (e.g. en). Default: auto-detect / config.",
)
@click.option(
    "--models-root",
    default=None,
    help="Download cache for models fetched by id (local or external drive).",
)
@click.option(
    "--model-path",
    default=None,
    help="Local model folder: Qwen3-ASR, MLX (config.json + weights) or "
         "CTranslate2 (model.bin). Its format picks the engine.",
)
@click.option(
    "--model-id",
    default=None,
    help="HuggingFace model id used when --model-path is unset or invalid.",
)
@click.option(
    "--engine",
    type=click.Choice(["auto", *ENGINES]),
    default=None,
    help="Backend: qwen or mlx (Apple Silicon GPU), or faster-whisper (CPU/CUDA, "
         "all platforms). Default auto: follows the local model's format, else "
         "the platform.",
)
@click.option(
    "--device",
    default=None,
    help="faster-whisper only: auto, cpu, or cuda. MLX always uses the Apple GPU.",
)
@click.option(
    "--compute-type",
    default=None,
    help="faster-whisper only: CTranslate2 compute type (int8, float16, "
         "int8_float16).",
)
@click.option(
    "--context",
    default=None,
    help="Domain terms biasing recognition, space-separated (qwen engine only). "
         "Names and jargon actually spoken; guessing adds nothing and risks "
         "the model inserting terms that were not said.",
)
@click.option(
    "--word-timestamps/--no-word-timestamps",
    "word_timestamps",
    default=None,
    help="Per-word timings. Costs an extra alignment pass; on by default only "
         "when the json format is requested.",
)
@click.option(
    "--diarize",
    is_flag=True,
    help="Label speakers (who spoke when) with pyannote. Requires the 'diarize' "
         "extra. A local pipeline folder in diarization.model_path needs no "
         "HuggingFace token.",
)
@click.option(
    "--diarize-only",
    is_flag=True,
    help="Add speaker labels to a saved recording's existing transcript without "
         "transcribing again. PATH is the recording (e.g. live-*.wav); its .json "
         "must sit beside it. Updates that session's files in place.",
)
@click.option(
    "--diarization-step",
    type=float,
    default=None,
    help="Seconds between diarization windows. Lower is slower and more "
         "precise; 2.0 is ~1.9x faster than pyannote's 1.0 default at 95.7% "
         "agreement, while 3.0 starts merging speakers.",
)
@click.option(
    "--speakers",
    type=int,
    default=None,
    help="Exact number of speakers, when known. Improves clustering.",
)
@click.option(
    "--min-speakers",
    type=int,
    default=None,
    help="Lower bound on the speaker count when the exact number is unknown.",
)
@click.option(
    "--max-speakers",
    type=int,
    default=None,
    help="Upper bound on the speaker count. Pair with --min-speakers to bound "
         "the search (e.g. 8-12) instead of guessing an exact number.",
)
@click.option(
    "--live",
    is_flag=True,
    help="Transcribe the microphone instead of a file. Prints each sentence as "
         "you pause. Ctrl+C to stop.",
)
@click.option(
    "--input-device",
    "input_device",
    default=None,
    help="Input device name or index for --live. Default: the system input.",
)
@click.option(
    "--list-devices",
    is_flag=True,
    help="List input devices and exit.",
)
@click.option(
    "--silence",
    type=float,
    default=0.5,
    show_default=True,
    help="Pause length that ends a sentence, in seconds (--live only).",
)
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    help="When PATH is a directory, search recursively for media files.",
)
@click.version_option(__version__, prog_name="transcribe")
def main(
    path: Path,
    output_dir: Path | None,
    formats: str,
    language: str | None,
    models_root: str | None,
    model_path: str | None,
    model_id: str | None,
    engine: str | None,
    device: str | None,
    compute_type: str | None,
    context: str | None,
    word_timestamps: bool | None,
    diarize: bool,
    diarize_only: bool,
    diarization_step: float | None,
    speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    live: bool,
    input_device: str | None,
    list_devices: bool,
    silence: float,
    recursive: bool,
) -> None:
    """Transcribe video or audio locally on the Apple Silicon GPU.

    PATH may be a media file or a directory of media files. With --live, PATH is
    omitted and the microphone is transcribed instead.
    """
    format_set = _parse_formats(formats)
    cfg = resolve_whisper_config(
        models_root=models_root,
        model_path=model_path,
        model_id=model_id,
        language=language,
        language_explicit=language is not None,
        engine=engine,
        device=device,
        compute_type=compute_type,
        context=context,
    )

    want_words = word_timestamps if word_timestamps is not None else ("json" in format_set)
    if diarize:
        want_words = True
    if speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise click.UsageError(
            "--speakers pins an exact count; use --min-speakers/--max-speakers instead, not both."
        )
    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers > max_speakers
    ):
        raise click.UsageError("--min-speakers cannot exceed --max-speakers.")
    diar_cfg = (
        resolve_diarization_config(
            step=diarization_step,
            speakers=speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        if diarize or diarize_only
        else None
    )

    if list_devices:
        from local_transcription_service.live import list_input_devices

        for index, name, is_default in list_input_devices():
            click.echo(f"  [{index}] {name}{'  <-- system input' if is_default else ''}")
        return

    if diarize_only:
        # Before the engine preflight: this path never runs speech recognition,
        # so it must not demand an ASR backend be installed.
        if live:
            raise click.UsageError("--diarize-only works on a saved recording, not with --live.")
        if path is None:
            raise click.UsageError("--diarize-only needs PATH: the recording to label.")
        _run_diarize_only(path, diar_cfg, output_dir, format_set, cfg)
        return

    if not live and path is None:
        raise click.UsageError("Missing argument 'PATH'. Use --live to read the microphone.")

    try:
        selected_engine = resolve_engine(cfg)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    model_ref, model_warning = resolve_model_ref(cfg, selected_engine)

    from local_transcription_service.whisper_engine import EngineUnavailable, _require

    try:
        _require(
            {"qwen": "mlx_qwen3_asr", ENGINE_MLX: "mlx_whisper"}.get(
                selected_engine, "faster_whisper"
            ),
            selected_engine,
        )
    except EngineUnavailable as exc:
        raise click.ClickException(str(exc)) from exc

    if live:
        # A live session always keeps a word-timed .json beside its .wav, so it can
        # be diarized later with --diarize-only instead of transcribed again. An
        # explicit --no-word-timestamps is still honoured.
        _run_live(cfg, model_ref, selected_engine, word_timestamps is not False,
                  input_device, silence, output_dir, format_set | {"json"}, diar_cfg)
        return

    try:
        media_files = collect_media_files(path, recursive=recursive)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Model root: {cfg.models_root}")
    click.echo(f"Model:      {model_ref}")
    if model_warning:
        # Never fall back silently: an ignored model_path otherwise shows up as
        # an unexplained multi-gigabyte download.
        click.echo(f"  warning:  {model_warning} — falling back to {model_ref}", err=True)
    if selected_engine == "qwen":
        detail = "mlx-qwen3-asr (Apple GPU)"
    elif selected_engine == ENGINE_MLX:
        detail = "mlx-whisper (Apple GPU)"
    else:
        detail = f"faster-whisper ({cfg.device}, {cfg.compute_type})"
    click.echo(f"Engine:     {detail}{', word timings' if want_words else ''}")
    if diar_cfg is not None:
        if diar_cfg.speakers:
            spk = f", exactly {diar_cfg.speakers} speakers"
        elif diar_cfg.min_speakers or diar_cfg.max_speakers:
            lo = diar_cfg.min_speakers or "?"
            hi = diar_cfg.max_speakers or "?"
            spk = f", {lo}-{hi} speakers"
        else:
            spk = ", speaker count auto-detected"
        click.echo(f"Diarize:    {diar_cfg.model_id} (step {diar_cfg.step}s{spk})")
    click.echo(f"Files:      {len(media_files)}")

    failures = 0
    for media in media_files:
        click.echo(f"\n→ {media}")
        wav: Path | None = None
        try:
            wav = extract_whisper_wav(media)
            result = transcribe_audio(
                wav,
                cfg,
                word_timestamps=want_words,
                show_progress=sys.stderr.isatty(),
            )
            if diar_cfg is not None:
                from local_transcription_service.diarize import (
                    apply_diarization,
                    diarize_audio,
                )

                turns = diarize_audio(
                    wav,
                    diar_cfg,
                    models_root=cfg.models_root,
                    show_progress=sys.stderr.isatty(),
                )
                result = apply_diarization(result, turns)
                found = sorted({t.speaker for t in turns})
                click.echo(f"  speakers: {len(found)} ({', '.join(found)})")

            stem_dir = (output_dir if output_dir is not None else Path.cwd()).resolve()
            stem_dir.mkdir(parents=True, exist_ok=True)
            stem = stem_dir / media.stem
            written = write_outputs(result, stem, format_set)
            lang = result.language or "unknown"
            click.echo(f"  language: {lang}")
            for out in written:
                click.echo(f"  wrote: {out}")
        except Exception as exc:
            failures += 1
            click.echo(f"  error: {exc}", err=True)
        finally:
            if wav is not None and wav.exists():
                try:
                    wav.unlink()
                except OSError:
                    pass

    if failures:
        raise click.ClickException(f"Failed on {failures} of {len(media_files)} file(s).")


if __name__ == "__main__":
    main(standalone_mode=True)
    sys.exit(0)
