import json
from pathlib import Path
from engine.types import Segment, TranscriptResult
from engine.output import fmt_timestamp, to_markdown, to_srt, to_json, write_outputs

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
