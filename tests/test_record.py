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

from engine.record import parse_device_listing

MACOS_LISTING = """[AVFoundation indev @ 0x7f8] AVFoundation video devices:
[AVFoundation indev @ 0x7f8] [0] FaceTime HD Camera
[AVFoundation indev @ 0x7f8] [1] Capture screen 0
[AVFoundation indev @ 0x7f8] AVFoundation audio devices:
[AVFoundation indev @ 0x7f8] [0] MacBook Pro Microphone
[AVFoundation indev @ 0x7f8] [1] BlackHole 2ch
"""

WINDOWS_LISTING = '''[dshow @ 000] "Integrated Camera" (video)
[dshow @ 000]   Alternative name "@device_pnp_\\\\?\\usb#vid"
[dshow @ 000] "Microphone (Realtek Audio)" (audio)
[dshow @ 000] "Stereo Mix" (audio)
'''

def test_macos_listing_excludes_video_devices():
    devs = parse_device_listing(MACOS_LISTING, "darwin")
    names = [d["name"] for d in devs]
    assert names == ["MacBook Pro Microphone", "BlackHole 2ch"]
    assert "FaceTime HD Camera" not in names   # cameras must not leak in

def test_macos_listing_uses_audio_index_ids():
    devs = parse_device_listing(MACOS_LISTING, "darwin")
    assert [d["id"] for d in devs] == [":0", ":1"]

def test_windows_dshow_listing_is_parsed():
    devs = parse_device_listing(WINDOWS_LISTING, "win32")
    assert [d["name"] for d in devs] == ["Microphone (Realtek Audio)", "Stereo Mix"]
    assert devs[0]["id"] == "audio=Microphone (Realtek Audio)"

def test_windows_listing_excludes_video():
    devs = parse_device_listing(WINDOWS_LISTING, "win32")
    assert all("Camera" not in d["name"] for d in devs)
