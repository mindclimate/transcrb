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


# ---- re-transcribing must not destroy the previous transcript -----------------
# Running a recording again with different settings used to overwrite the first
# result in place, so there was no way to compare the two — and no warning that
# the earlier one was gone.

import os
from engine.output import ARCHIVE_DIRNAME

def _archives(out, name):
    root = out / name / ARCHIVE_DIRNAME
    return sorted(p.name for p in root.iterdir()) if root.exists() else []

def test_a_first_run_archives_nothing(tmp_path):
    out = tmp_path / "out"
    write_outputs(_sample(), out, "meeting-x")
    assert _archives(out, "meeting-x") == []

def test_re_running_keeps_the_previous_transcript(tmp_path):
    out = tmp_path / "out"
    write_outputs(_sample(), out, "meeting-x")
    first = (out / "meeting-x" / "transcript.md").read_text(encoding="utf-8")
    # A second run with a different result, as a different speaker count gives.
    write_outputs(_single_result(), out, "meeting-x")

    kept = _archives(out, "meeting-x")
    assert len(kept) == 1
    archived = out / "meeting-x" / ARCHIVE_DIRNAME / kept[0]
    assert archived.joinpath("transcript.md").read_text(encoding="utf-8") == first
    # every artefact of the old run travels together
    assert {p.name for p in archived.iterdir()} == {
        "transcript.md", "transcript.srt", "transcript.json"}

def test_the_canonical_paths_always_hold_the_newest_run(tmp_path):
    """Downstream ingestion reads transcript.md; it must not have to look."""
    out = tmp_path / "out"
    write_outputs(_sample(), out, "meeting-x")
    written = write_outputs(_single_result(), out, "meeting-x")
    assert written["md"] == out / "meeting-x" / "transcript.md"
    assert "hello" in written["md"].read_text(encoding="utf-8")

def test_a_third_run_keeps_both_earlier_ones(tmp_path):
    out = tmp_path / "out"
    for i in range(3):
        write_outputs(_sample(), out, "meeting-x")
        # Distinct mtimes: archives are labelled by when the run was produced.
        stamp = 1_700_000_000 + i * 3600
        for f in (out / "meeting-x").glob("transcript.*"):
            os.utime(f, (stamp, stamp))
    assert len(_archives(out, "meeting-x")) == 2

def test_two_runs_sharing_a_timestamp_do_not_collide(tmp_path):
    out = tmp_path / "out"
    for _ in range(3):
        write_outputs(_sample(), out, "meeting-x")
        for f in (out / "meeting-x").glob("transcript.*"):
            os.utime(f, (1_700_000_000, 1_700_000_000))   # identical every time
    kept = _archives(out, "meeting-x")
    assert len(kept) == 2 and len(set(kept)) == 2

def test_words_json_is_archived_with_its_transcript(tmp_path):
    out = tmp_path / "out"
    from engine.types import Word
    with_words = TranscriptResult(
        language="en", model="m", duration=1.0,
        segments=[Segment(0.0, 1.0, "SPEAKER_00", "hi")],
        words=[Word(0.0, 1.0, "hi", lang="en")])
    write_outputs(with_words, out, "meeting-x")
    write_outputs(with_words, out, "meeting-x")
    archived = out / "meeting-x" / ARCHIVE_DIRNAME / _archives(out, "meeting-x")[0]
    assert archived.joinpath("words.json").is_file()

def test_the_archive_is_never_swept_into_a_later_archive(tmp_path):
    out = tmp_path / "out"
    for _ in range(3):
        write_outputs(_sample(), out, "meeting-x")
    root = out / "meeting-x" / ARCHIVE_DIRNAME
    for archived in root.iterdir():
        assert not (archived / ARCHIVE_DIRNAME).exists()

def test_the_inbox_copy_is_replaced_not_archived(tmp_path):
    """The inbox is a drop point that gets consumed; duplicates there would be
    ingested twice."""
    out, inbox = tmp_path / "out", tmp_path / "inbox"
    write_outputs(_sample(), out, "meeting-x", inbox=inbox)
    write_outputs(_single_result(), out, "meeting-x", inbox=inbox)
    assert sorted(p.name for p in inbox.iterdir()) == ["meeting-x.json", "meeting-x.md"]
    assert "hello" in (inbox / "meeting-x.md").read_text(encoding="utf-8")
