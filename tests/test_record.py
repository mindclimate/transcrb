import struct
from pathlib import Path

import numpy as np
import soundfile as sf

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


# ---- level metering ----------------------------------------------------------
# The standup recording was 24 minutes of digital silence: BlackHole was picked
# as the input while macOS was still sending call audio to the headphones, so
# nothing was ever routed into it. ffmpeg reports success for that, the file is
# a healthy 46MB, and Whisper turns the silence into "you" on repeat. Levels are
# the only signal that tells the difference.

from engine.record import (
    SILENCE_DBFS,
    SILENCE_FLOOR_DBFS,
    build_probe_command,
    is_silent,
    peak_dbfs,
    silence_hint,
)

def _write_wav(path: Path, samples, rate: int = 16000) -> Path:
    sf.write(str(path), np.asarray(samples, dtype="float32"), rate, subtype="PCM_16")
    return path

def _tone(seconds: float, amplitude: float, rate: int = 16000):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return amplitude * np.sin(2 * np.pi * 440 * t)

def test_peak_dbfs_reports_the_floor_for_digital_silence(tmp_path):
    wav = _write_wav(tmp_path / "silent.wav", np.zeros(16000))
    assert peak_dbfs(wav) == SILENCE_FLOOR_DBFS

def test_peak_dbfs_measures_a_half_scale_tone_at_about_minus_six(tmp_path):
    wav = _write_wav(tmp_path / "tone.wav", _tone(1.0, 0.5))
    assert -7.0 < peak_dbfs(wav) < -5.0

def test_is_silent_splits_the_dead_recording_from_the_working_mic():
    assert is_silent(SILENCE_FLOOR_DBFS)
    assert is_silent(-91.0)          # the 24-minute standup recording
    assert not is_silent(-33.8)      # the built-in mic, captured for comparison
    assert not is_silent(SILENCE_DBFS + 1.0)

def test_peak_dbfs_tail_only_measures_the_end_of_the_file(tmp_path):
    # Loud open, then the routing dies: the tail must read as silence so the
    # live meter can catch output switching away mid-meeting.
    wav = _write_wav(tmp_path / "died.wav",
                     np.concatenate([_tone(1.0, 0.5), np.zeros(16000 * 3)]))
    assert not is_silent(peak_dbfs(wav))
    assert is_silent(peak_dbfs(wav, tail_seconds=2.0))

def _wav_with_metadata(path: Path, pcm: bytes, declared_size: int | None = None) -> Path:
    """A wav shaped like ffmpeg's: a LIST/INFO chunk sits before the samples."""
    fmt = struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
    info = b"INFOISFT" + struct.pack("<I", 8) + b"Lavf62.1"
    size = len(pcm) if declared_size is None else declared_size
    body = (b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"LIST" + struct.pack("<I", len(info)) + info
            + b"data" + struct.pack("<I", size) + pcm)
    path.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)
    return path

def test_peak_dbfs_skips_the_metadata_chunk_ffmpeg_writes(tmp_path):
    # Reading from a fixed 44-byte offset would treat "LIST...Lavf" as samples
    # and report a bogus level for an otherwise silent file.
    wav = _wav_with_metadata(tmp_path / "meta.wav", b"\x00\x00" * 16000)
    assert peak_dbfs(wav) == SILENCE_FLOOR_DBFS

def test_peak_dbfs_reads_a_file_still_being_recorded(tmp_path):
    # ffmpeg leaves the data-chunk size at 0 until it closes the file, so the
    # live meter has to trust the bytes on disk rather than the header.
    pcm = (np.asarray(_tone(1.0, 0.5) * 32767, dtype="<i2")).tobytes()
    wav = _wav_with_metadata(tmp_path / "growing.wav", pcm, declared_size=0)
    assert -7.0 < peak_dbfs(wav) < -5.0

def test_peak_dbfs_treats_a_headers_only_file_as_silent(tmp_path):
    wav = _wav_with_metadata(tmp_path / "empty.wav", b"", declared_size=0)
    assert peak_dbfs(wav) == SILENCE_FLOOR_DBFS

def test_probe_command_captures_a_short_sample_from_the_device(tmp_path):
    cmd = build_probe_command(":1", tmp_path / "probe.wav", "darwin", 1.5)
    assert cmd[0] == "ffmpeg"
    assert "avfoundation" in cmd and ":1" in cmd
    assert cmd[cmd.index("-t") + 1] == "1.5"

def test_silence_hint_names_the_multi_output_fix_for_blackhole():
    hint = silence_hint("BlackHole 2ch")
    assert "Multi-Output" in hint          # the actual fix, not just "no signal"
    assert "BlackHole" in hint

def test_silence_hint_falls_back_to_mic_advice_for_a_real_input():
    assert "Multi-Output" not in silence_hint("MacBook Pro Microphone")

def test_record_command_flushes_so_the_live_meter_can_read_the_file():
    # Without this ffmpeg buffers the entire recording and the file stays 0
    # bytes on disk until it closes, so mid-recording metering sees nothing.
    cmd = build_record_command(":1", Path("/tmp/r.wav"), "darwin")
    assert cmd[cmd.index("-flush_packets") + 1] == "1"
