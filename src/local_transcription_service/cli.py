"""Click CLI entry point: ``transcribe``."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from local_transcription_service import __version__
from local_transcription_service.config import resolve_whisper_config
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
    help="Path to a local CTranslate2 model folder containing model.bin.",
)
@click.option(
    "--model-id",
    default=None,
    help="HuggingFace faster-whisper model id when model-path is unset/invalid.",
)
@click.option(
    "--device",
    default=None,
    help="Device: auto, cpu, or cuda.",
)
@click.option(
    "--compute-type",
    default=None,
    help="CTranslate2 compute type (e.g. int8, float16, int8_float16).",
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
    device: str | None,
    compute_type: str | None,
    recursive: bool,
) -> None:
    """Transcribe video or audio with local faster-whisper.

    PATH may be a media file or a directory of media files.
    """
    format_set = _parse_formats(formats)
    cfg = resolve_whisper_config(
        models_root=models_root,
        model_path=model_path,
        model_id=model_id,
        device=device,
        compute_type=compute_type,
        language=language,
        language_explicit=language is not None,
    )

    try:
        media_files = collect_media_files(path, recursive=recursive)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Model root: {cfg.models_root}")
    if cfg.model_path:
        click.echo(f"Model path: {cfg.model_path}")
    else:
        click.echo(f"Model id:   {cfg.model_id}")
    click.echo(f"Device:     {cfg.device} ({cfg.compute_type})")
    click.echo(f"Files:      {len(media_files)}")

    failures = 0
    for media in media_files:
        click.echo(f"\n→ {media}")
        wav: Path | None = None
        try:
            wav = extract_whisper_wav(media)
            result = transcribe_audio(wav, cfg)
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
