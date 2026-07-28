import json
from pathlib import Path
from engine.types import Segment, TranscriptResult
from engine.output import fmt_timestamp, to_markdown, to_srt, to_json, write_outputs, _srt_time

def _sample():
    return TranscriptResult(
        language="he", model="ivrit-ai/whisper-large-v3", duration=3.5,
        segments=[
            Segment(0.0, 2.0, "SPEAKER_00", "שלום לכולם"),
            Segment(2.5, 3.5, "SPEAKER_01", "Hi there"),
        ],
    )

def test_fmt_timestamp():
    assert fmt_timestamp(0) == "00:00:00"
    assert fmt_timestamp(3661) == "01:01:01"

def test_markdown_preserves_rtl_and_labels():
    md = to_markdown(_sample())
    assert "שלום לכולם" in md            # RTL text verbatim
    assert "SPEAKER_00" in md
    assert "[00:00:00]" in md

def test_srt_indices_and_arrow():
    srt = to_srt(_sample())
    assert "1\n00:00:00,000 --> 00:00:02,000" in srt
    assert "2\n" in srt

def test_json_roundtrip():
    data = json.loads(to_json(_sample()))
    assert data["language"] == "he"
    assert data["segments"][0]["text"] == "שלום לכולם"

def test_srt_time_no_ms_overflow():
    result = TranscriptResult(
        language="en", model="large-v3", duration=2.0,
        segments=[Segment(0.0, 1.9999995, "SPEAKER_00", "hi")],
    )
    srt = to_srt(result)
    assert "00:00:02,000" in srt
    assert ",1000" not in srt
    assert _srt_time(1.9999995) == "00:00:02,000"

def test_write_outputs_creates_files_and_inbox_copy(tmp_path):
    out = tmp_path / "out"
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    written = write_outputs(_sample(), out, "meeting-x", inbox=inbox)
    assert (out / "meeting-x" / "transcript.md").exists()
    assert (out / "meeting-x" / "transcript.srt").exists()
    assert (out / "meeting-x" / "transcript.json").exists()
    assert (inbox / "meeting-x.md").exists()
    assert (inbox / "meeting-x.json").exists()
    assert written["md"] == out / "meeting-x" / "transcript.md"


# ---- mixed-language transcripts ----------------------------------------------

def _mixed_result():
    return TranscriptResult(
        language="en+he", model="large-v3+models/ivrit-whisper-ct2", duration=90.0,
        segments=[Segment(0.0, 30.0, "SPEAKER_00", "hello everyone", lang="en"),
                  Segment(30.0, 60.0, "SPEAKER_01", "שלום לכולם", lang="he")])

def _single_result():
    return TranscriptResult(
        language="en", model="large-v3", duration=10.0,
        segments=[Segment(0.0, 5.0, "SPEAKER_00", "hello", lang="en")])

def test_markdown_tags_each_line_with_its_language_when_mixed():
    md = to_markdown(_mixed_result())
    assert "SPEAKER_00 (en): hello everyone" in md
    assert "SPEAKER_01 (he): שלום לכולם" in md

def test_markdown_leaves_single_language_transcripts_untagged():
    # Tagging every line of a normal transcript would be noise.
    md = to_markdown(_single_result())
    assert "SPEAKER_00: hello" in md
    assert "(en)" not in md.split("- Language")[1].split("\n", 1)[1]

def test_json_records_the_language_of_every_segment():
    import json
    data = json.loads(to_json(_mixed_result()))
    assert [s["lang"] for s in data["segments"]] == ["en", "he"]

def test_srt_stays_plain_text_for_players():
    srt = to_srt(_mixed_result())
    assert "(en)" not in srt and "(he)" not in srt
    assert "שלום לכולם" in srt
