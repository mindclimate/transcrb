from pathlib import Path
import cli
from engine.types import TranscriptResult, Segment

def test_cli_transcribes_single_file(monkeypatch, tmp_path, capsys):
    audio = tmp_path / "call.wav"; audio.write_bytes(b"x")
    fake = TranscriptResult(language="en", model="large-v3", duration=1.0,
                            segments=[Segment(0.0, 1.0, "SPEAKER_00", "hi")])
    monkeypatch.setattr(cli, "transcribe_file", lambda *a, **k: fake)
    rc = cli.main([str(audio), "--out", str(tmp_path / "out"), "--lang", "en"])
    assert rc == 0
    assert (tmp_path / "out" / "call" / "transcript.md").exists()
    out = capsys.readouterr().out
    assert "transcript.md" in out

def test_cli_errors_on_missing_path(tmp_path):
    rc = cli.main([str(tmp_path / "nope.wav")])
    assert rc == 1
