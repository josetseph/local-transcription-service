"""Click CLI entry point: ``transcribe``."""

from __future__ import annotations

import sys
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


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("path", type=click.Path(exists=True, path_type=Path))
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
    help="Directory for Whisper model downloads/cache (local or external drive).",
)
@click.option(
    "--model-path",
    default=None,
    help="Path to a local MLX model folder (config.json + weights.safetensors).",
)
@click.option(
    "--model-id",
    default=None,
    help="HuggingFace MLX model id when model-path is unset/invalid.",
)
@click.option(
    "--engine",
    type=click.Choice(["auto", *ENGINES]),
    default=None,
    help="Backend: mlx (Apple Silicon GPU) or faster-whisper (CPU/CUDA, all "
         "platforms). Default auto: follows the local model's format, else the "
         "platform.",
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
    "--word-timestamps/--no-word-timestamps",
    "word_timestamps",
    default=None,
    help="Per-word timings. Costs an extra alignment pass; on by default only "
         "when the json format is requested.",
)
@click.option(
    "--diarize",
    is_flag=True,
    help="Label speakers (who spoke when) with pyannote. Requires the "
         "'diarize' extra and a HuggingFace token for the gated model.",
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
    word_timestamps: bool | None,
    diarize: bool,
    diarization_step: float | None,
    speakers: int | None,
    recursive: bool,
) -> None:
    """Transcribe video or audio locally on the Apple Silicon GPU (MLX).

    PATH may be a media file or a directory of media files.
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
    )

    want_words = word_timestamps if word_timestamps is not None else ("json" in format_set)
    if diarize:
        want_words = True
    diar_cfg = (
        resolve_diarization_config(step=diarization_step, speakers=speakers)
        if diarize
        else None
    )

    try:
        selected_engine = resolve_engine(cfg)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    model_ref, model_warning = resolve_model_ref(cfg, selected_engine)

    from local_transcription_service.whisper_engine import EngineUnavailable, _require

    try:
        _require("mlx_whisper" if selected_engine == ENGINE_MLX else "faster_whisper",
                 selected_engine)
    except EngineUnavailable as exc:
        raise click.ClickException(str(exc)) from exc

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
    if selected_engine == ENGINE_MLX:
        detail = "mlx-whisper (Apple GPU)"
    else:
        detail = f"faster-whisper ({cfg.device}, {cfg.compute_type})"
    click.echo(f"Engine:     {detail}{', word timings' if want_words else ''}")
    if diar_cfg is not None:
        spk = f", {diar_cfg.speakers} speakers" if diar_cfg.speakers else ""
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

                turns = diarize_audio(wav, diar_cfg, models_root=cfg.models_root)
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
