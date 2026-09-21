"""Summarize a transcript with a local GGUF chat model, in-process via llama.cpp.

One model in memory at a time: this runs after speech recognition and
diarization have released theirs, and releases its own when done.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from local_transcription_service.memory import release_accelerator_memory
from local_transcription_service.whisper_engine import TranscriptResult
from local_transcription_service.writers import speaker_names

DEFAULT_N_CTX = 32768
# Left free in the window for the summary itself. A finished summary is a few
# hundred words; this is the floor below which the input is split instead.
MIN_OUTPUT_TOKENS = 2048

SHAPE = """# <title of three to six words>

## Summary
### Flow Summary
<one or two sentences: what kind of session this was and what it covered>

### <topic heading>
- <point>
    - <sub-point, only where one is needed>

### Next Steps
- (Speaker N) <something that speaker said they or others will do>

### Decisions Made
- <decision>"""

RULES = """Rules:
- Write three to five topic sections, in the order the topics came up.
- Use only what the source says. Add no facts, names or figures of your own.
- Copy numbers, amounts, dates and names exactly as they appear.
- The source comes from automatic speech recognition, which mishears words. \
Where a phrase makes no sense in context, leave that point out rather than guess.
- Bullets are short fragments, not full sentences.
- If no next steps or no decisions were stated, write "- None stated." under that heading.
- Output the markdown only: no preamble, no code fence."""

SYSTEM = (
    "You write the notes for a recording. Output markdown in exactly this shape:\n\n"
    f"{SHAPE}\n\n{RULES}"
)

PART_SYSTEM = (
    "You are given one part of a long transcript. List, as plain bullets and in order, "
    "every topic covered, every fact, figure and name, every decision, and every action "
    "someone said they will take, with the speaker label of who said it. Use only what "
    "the transcript says; where a phrase makes no sense, leave it out rather than guess."
)


@dataclass
class SummaryConfig:
    model_path: Path
    n_ctx: int = DEFAULT_N_CTX


def transcript_lines(result: TranscriptResult) -> list[str]:
    """Speaker-labelled lines, consecutive turns by one speaker merged.

    No timestamps: they cost about a quarter of the tokens and the summary has
    no use for them. Labels match the md transcript, so "(Speaker 1)" in the
    summary points at the same person.
    """
    names = speaker_names(result.segments)
    lines: list[str] = []
    last: str | None = None
    for seg in result.segments:
        text = seg.text.strip()
        if not text:
            continue
        # Bounded, so a one-speaker lecture still splits across windows.
        if lines and seg.speaker == last and len(lines[-1]) < 1000:
            lines[-1] += " " + text
        else:
            lines.append(f"{names[seg.speaker]}: {text}" if seg.speaker else text)
            last = seg.speaker
    return lines or [result.text.strip()]


def split_lines(lines: list[str], sizes: list[int], budget: int) -> list[str]:
    """Group lines into pieces of at most ``budget`` tokens, never cutting a line."""
    pieces: list[str] = []
    current: list[str] = []
    used = 0
    for line, size in zip(lines, sizes):
        if current and used + size > budget:
            pieces.append("\n".join(current))
            current, used = [], 0
        current.append(line)
        used += size
    if current:
        pieces.append("\n".join(current))
    return pieces


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    return text.strip()


def summarize(result: TranscriptResult, cfg: SummaryConfig) -> str:
    """Return the ``# title`` + ``## Summary`` markdown for a transcript."""
    from llama_cpp import Llama

    llm = None
    try:
        # Flash attention and the compact sliding-window cache are what let a
        # 32k window fit beside a 4B Gemma in 24 GB.
        llm = Llama(model_path=str(cfg.model_path), n_ctx=cfg.n_ctx, n_gpu_layers=-1,
                    flash_attn=True, swa_full=False, verbose=False)

        def count(text: str) -> int:
            return len(llm.tokenize(text.encode("utf-8"), add_bos=False)) + 1

        def ask(system: str, user: str) -> str:
            # The answer gets whatever the prompt leaves, not a fixed cap.
            room = cfg.n_ctx - count(system) - count(user) - 64
            if room < 256:
                raise RuntimeError(
                    f"prompt leaves {room} tokens of a {cfg.n_ctx}-token window; "
                    "raise summary.n_ctx"
                )
            out = llm.create_chat_completion(
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.2, repeat_penalty=1.12, max_tokens=room,
            )
            return _strip_fence(out["choices"][0]["message"]["content"] or "")

        lines = transcript_lines(result)
        budget = cfg.n_ctx - count(SYSTEM) - MIN_OUTPUT_TOKENS
        pieces = split_lines(lines, [count(line) for line in lines], budget)
        if len(pieces) == 1:
            return ask(SYSTEM, "TRANSCRIPT:\n" + pieces[0])
        # Too long for one window: notes per part, then the summary from the notes.
        notes = [
            ask(PART_SYSTEM, f"PART {i} OF {len(pieces)}:\n{piece}")
            for i, piece in enumerate(pieces, start=1)
        ]
        return ask(SYSTEM, "NOTES ON THE RECORDING, IN ORDER:\n" + "\n\n".join(notes))
    finally:
        if llm is not None:
            llm.close()
            del llm
        release_accelerator_memory()
