from pathlib import Path
from engine.record import build_record_command

def test_macos_uses_avfoundation():
    cmd = build_record_command(":0", Path("/tmp/r.wav"), "darwin")
    assert "ffmpeg" == cmd[0]
    assert "avfoundation" in cmd
    assert ":0" in cmd
    assert cmd[-1] == "/tmp/r.wav"

def test_windows_uses_dshow():
    cmd = build_record_command("audio=Microphone", Path("C:/r.wav"), "win32")
    assert "dshow" in cmd
    assert "audio=Microphone" in cmd

def test_forces_16k_mono():
    cmd = build_record_command(":0", Path("/tmp/r.wav"), "darwin")
    assert "16000" in cmd
    assert "1" in cmd  # mono channel count
