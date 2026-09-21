from local_transcription_service.summarize import split_lines, transcript_lines
from local_transcription_service.whisper_engine import Segment, TranscriptResult
from local_transcription_service.writers import format_md

SEGMENTS = [
    Segment("Hi.", 2.0, 3.0, speaker="SPEAKER_01"),
    Segment("There.", 3.0, 4.0, speaker="SPEAKER_01"),
    Segment("Yes.", 5040.0, 5041.0, speaker="SPEAKER_00"),
    Segment("", 5045.0, 5046.0, speaker="SPEAKER_00"),
]


def test_md_numbers_speakers_by_first_appearance_and_keeps_minutes_past_an_hour():
    md = format_md(SEGMENTS, "lecture")
    assert md.startswith("# lecture\n\n## Transcript\n[00:02] Speaker 1: Hi. There.\n")   # short cues join
    assert "[84:00] Speaker 2: Yes." in md
    assert "[84:05]" not in md                           # empty segment dropped
    assert format_md(SEGMENTS, "x", "# T\n\n## Summary\n").startswith("# T\n\n## Summary\n\n## Transcript")
    assert "[00:01] plain" in format_md([Segment("plain", 1.0, 2.0)], "x")


def test_summary_input_uses_the_same_speaker_names_and_splits_on_lines():
    lines = transcript_lines(TranscriptResult(text="", language="en", segments=SEGMENTS))
    assert lines == ["Speaker 1: Hi. There.", "Speaker 2: Yes."]
    assert split_lines(["a", "b", "c"], [5, 5, 5], 10) == ["a\nb", "c"]
    assert split_lines(["big"], [50], 10) == ["big"]        # never cuts a line


def test_punctuation_survives_an_aligner_word_count_mismatch():
    from local_transcription_service.whisper_engine import WordTiming, _restore_punctuation

    words = [WordTiming(w, 0.0, 0.0) for w in ["so", "yes", "uh", "okay", "next"]]
    _restore_punctuation(words, "So, yes. Okay, next.".split())      # aligner heard an extra "uh"
    assert [w.word for w in words] == ["So,", "yes.", "uh", "Okay,", "next."]
