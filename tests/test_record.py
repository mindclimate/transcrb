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
    # On a Mac that has built the native recorder the probe goes through it, so
    # this asserts the shape of whichever one is actually going to record.
    cmd = build_probe_command(":1", tmp_path / "probe.wav", "darwin", 1.5)
    if cmd[0] == "ffmpeg":
        assert "avfoundation" in cmd and ":1" in cmd
        assert cmd[cmd.index("-t") + 1] == "1.5"
    else:
        assert cmd[cmd.index("--device-index") + 1] == "1"
        assert cmd[cmd.index("--seconds") + 1] == "1.5"

def test_probe_goes_through_the_recorder_that_will_do_the_meeting(tmp_path):
    # A microphone permission granted to one binary and refused to the other
    # would let the level check pass and then record an hour of silence.
    windows = build_probe_command("audio=Mic", tmp_path / "p.wav", "win32", 1.5)
    assert windows[0] == "ffmpeg"
    if native_recorder("darwin") is not None:
        assert build_probe_command(":1", tmp_path / "p.wav", "darwin", 1.5)[0] != "ffmpeg"

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


# ---- dropped audio -----------------------------------------------------------
# Two meetings were lost to this. ffmpeg's avfoundation input holds exactly one
# pending audio buffer and blocks the capture callback until it is read;
# CoreAudio cannot wait, so anything it produced during the stall is discarded
# before ffmpeg sees it — no error, no gap. The samples that survive are
# written back to back, so 40 minutes of meeting landed as 8
# minutes of speech at five times speed and Whisper returned "Thank you." on
# loop. Levels cannot catch it: the file was loud the whole way through.

def test_record_command_keeps_the_timeline_when_the_capture_drops_buffers():
    # aresample honours the input timestamps: audio the capture layer lost comes
    # back as silence in the right place, instead of every later word sliding
    # earlier. first_pts=0 is required with it — avfoundation timestamps start
    # at the host uptime (45093s in the measured case), not at zero.
    cmd = build_record_command(":1", Path("/tmp/r.wav"), "darwin")
    assert cmd[cmd.index("-af") + 1] == "aresample=async=1:first_pts=0"

from engine.record import DROPPED_FRACTION_LIMIT, dropped_fraction, dropped_hint

def test_dropped_fraction_is_zero_for_a_recording_that_captured_everything(tmp_path):
    wav = _write_wav(tmp_path / "clean.wav", _tone(4.0, 0.5))
    assert dropped_fraction(wav) == 0.0

def test_dropped_fraction_measures_the_silence_the_filler_wrote(tmp_path):
    # Half tone, half filled-in silence: the shape of a throttled capture.
    wav = _write_wav(tmp_path / "half.wav",
                     np.concatenate([_tone(2.0, 0.5), np.zeros(16000 * 2)]))
    assert 0.45 < dropped_fraction(wav) < 0.55

def test_dropped_fraction_ignores_the_zero_crossings_of_ordinary_audio(tmp_path):
    # A sine wave crosses zero 880 times a second. Counting bare zero samples
    # would call a perfect recording 0.1% dropped and, on quieter speech, far
    # more — only runs long enough to be a lost buffer count.
    wav = _write_wav(tmp_path / "quiet.wav", _tone(4.0, 0.002))
    assert dropped_fraction(wav) < 0.01

def test_dropped_hint_names_the_throttle_rather_than_the_symptom():
    hint = dropped_hint(0.73)
    assert "73%" in hint
    assert "ProcessType" in hint          # the one thing that actually fixes it

def test_dropped_hint_stops_blaming_the_throttle_once_the_native_recorder_is_used():
    # The native recorder is immune to the throttle — measured 0.1% loss under
    # taskpolicy -b against ffmpeg's 88.7% on the same mic. Telling someone to
    # check ProcessType when that is not what failed sends them at the wrong
    # thing entirely.
    hint = dropped_hint(0.73, recorder="native")
    assert "73%" in hint
    assert "ProcessType" not in hint


# ---- the native recorder -----------------------------------------------------
# ffmpeg's avfoundation input cannot be made reliable: one pending buffer, a
# blocking callback, about 10ms of tolerance. native/capture.swift replaces it
# with a thirty-second ring buffer and a separate writer thread. Measured on
# this machine under `taskpolicy -b`, the throttle that destroyed two meetings:
# ffmpeg lost 88.7% of a 30s capture, the native recorder lost 0.1%.

from engine.record import (
    NATIVE_RECORDER,
    build_native_command,
    native_recorder,
    parse_capture_summary,
    progress_path,
    read_progress,
    reported_dropped,
)

def test_the_native_recorder_ships_with_its_source_so_it_can_be_rebuilt():
    assert NATIVE_RECORDER.with_name("capture.swift").is_file()

def test_native_recorder_is_only_offered_on_macos(tmp_path):
    binary = tmp_path / "transcrb-capture"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    assert native_recorder("darwin", binary) == binary
    assert native_recorder("win32", binary) is None

def test_native_recorder_is_absent_until_it_has_been_built(tmp_path):
    assert native_recorder("darwin", tmp_path / "not-built") is None

def test_native_recorder_is_ignored_if_it_is_not_executable(tmp_path):
    # A source checkout has capture.swift but no binary; a failed build can
    # leave a zero-byte file behind. Neither may be handed a meeting.
    binary = tmp_path / "transcrb-capture"
    binary.write_text("")
    binary.chmod(0o644)
    assert native_recorder("darwin", binary) is None

def test_native_command_addresses_the_device_ffmpeg_would(tmp_path):
    # The UI hands back avfoundation's ":N", so the native recorder resolves the
    # same index through the same AVCaptureDevice list rather than guessing at
    # CoreAudio's ordering, which is a different list in a different order.
    cmd = build_native_command(":2", tmp_path / "r.wav", tmp_path / "cap")
    assert cmd[cmd.index("--device-index") + 1] == "2"
    assert cmd[cmd.index("--out") + 1] == str(tmp_path / "r.wav")

def test_native_command_asks_for_progress_while_the_meeting_runs(tmp_path):
    out = tmp_path / "r.wav"
    cmd = build_native_command(":2", out, tmp_path / "cap", progress=progress_path(out))
    assert cmd[cmd.index("--progress") + 1] == str(tmp_path / "r.progress.json")

def test_progress_reports_loss_before_the_recording_is_over(tmp_path):
    out = tmp_path / "r.wav"
    progress_path(out).write_text(
        '{"captured_seconds":120.0,"dropped":0.42,"reason":"recording"}')
    assert read_progress(out)["dropped"] == 0.42

def test_progress_is_empty_rather_than_raising_before_the_first_second(tmp_path):
    assert read_progress(tmp_path / "r.wav") == {}

def test_progress_survives_being_read_mid_write(tmp_path):
    # The file is replaced by rename so this should not happen, but a truncated
    # read must never take a recording down.
    out = tmp_path / "r.wav"
    progress_path(out).write_text('{"captured_seconds":12')
    assert read_progress(out) == {}

def test_capture_summary_is_the_last_json_the_recorder_printed():
    raw = b'starting\n{"dropped":0.0,"captured_seconds":63.0}\n'
    assert parse_capture_summary(raw)["captured_seconds"] == 63.0

def test_capture_summary_is_empty_for_a_recorder_that_reports_nothing():
    # ffmpeg prints no summary, and its stdout is not even captured.
    assert parse_capture_summary(None) == {}
    assert parse_capture_summary(b"") == {}
    assert parse_capture_summary(b"ffmpeg version 7.1\nsize=  1024kB\n") == {}

def test_loss_comes_from_the_recorders_own_count_when_it_kept_one(tmp_path):
    # The native recorder knows exactly how many frames CoreAudio never handed
    # it. Counting filled-in silence instead would also count a quiet room.
    wav = _write_wav(tmp_path / "quiet.wav", np.zeros(16000 * 4))
    assert reported_dropped({"dropped": 0.01}, wav) == 0.01

def test_loss_falls_back_to_measuring_silence_when_there_is_no_count(tmp_path):
    wav = _write_wav(tmp_path / "half.wav",
                     np.concatenate([_tone(2.0, 0.5), np.zeros(16000 * 2)]))
    assert 0.45 < reported_dropped({}, wav) < 0.55

def test_a_nonsense_count_does_not_pass_through_as_a_percentage(tmp_path):
    wav = _write_wav(tmp_path / "clean.wav", _tone(2.0, 0.5))
    assert reported_dropped({"dropped": "nonsense"}, wav) == 0.0
    assert reported_dropped({"dropped": 4.2}, wav) == 1.0


# ---- recordings a crash left behind ------------------------------------------
# The service being reinstalled mid-meeting on 3 Aug ended a recording without
# anyone stopping it. The 93MB of audio was on disk the whole time and nothing
# pointed at it, so it sat unnoticed for two days.

from engine.record import stalled_hint

def test_stalled_hint_names_the_dead_device_not_the_throttle():
    hint = stalled_hint(41.0, "AirPods Pro")
    assert "41 seconds" in hint
    assert "AirPods Pro" in hint
    assert "ProcessType" not in hint

def test_stalled_hint_works_without_a_device_name():
    assert "the input" in stalled_hint(12.0)


from engine.record import repair_wav_header, unfinished_recordings

def test_an_interrupted_recording_claims_to_be_empty_until_it_is_repaired(tmp_path):
    pcm = (np.asarray(_tone(2.0, 0.5) * 32767, dtype="<i2")).tobytes()
    wav = _wav_with_metadata(tmp_path / "cut.wav", pcm, declared_size=0)
    assert sf.info(str(wav)).frames == 0        # QuickTime and Finder agree
    assert repair_wav_header(wav) is True
    assert sf.info(str(wav)).frames == 32000

def test_repairing_a_finished_recording_changes_nothing(tmp_path):
    wav = _write_wav(tmp_path / "whole.wav", _tone(1.0, 0.5))
    before = wav.read_bytes()
    assert repair_wav_header(wav) is False
    assert wav.read_bytes() == before

def test_repairing_something_that_is_not_a_wav_does_not_raise(tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_bytes(b"this is not audio at all")
    assert repair_wav_header(junk) is False

def test_a_recording_with_no_transcript_is_reported_as_unfinished(tmp_path):
    recordings, out = tmp_path / "recordings", tmp_path
    recordings.mkdir()
    _write_wav(recordings / "recording-20260803-120243.wav", _tone(3.0, 0.5))
    _write_wav(recordings / "recording-20260803-173629.wav", _tone(3.0, 0.5))
    done = out / "recording-20260803-173629"
    done.mkdir()
    (done / "transcript.md").write_text("# Transcript")
    found = unfinished_recordings(recordings, out)
    assert [f["name"] for f in found] == ["recording-20260803-120243"]
    assert found[0]["seconds"] == 3.0

def test_unfinished_lists_the_newest_first(tmp_path):
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    for stamp in ("20260801-090000", "20260805-090000", "20260803-090000"):
        _write_wav(recordings / f"recording-{stamp}.wav", _tone(1.0, 0.5))
    names = [f["name"] for f in unfinished_recordings(recordings, tmp_path)]
    assert names == ["recording-20260805-090000", "recording-20260803-090000",
                     "recording-20260801-090000"]

def test_unfinished_ignores_a_recording_that_is_only_a_header(tmp_path):
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    _wav_with_metadata(recordings / "recording-empty.wav", b"", declared_size=0)
    assert unfinished_recordings(recordings, tmp_path) == []

def test_unfinished_is_empty_before_anything_has_been_recorded(tmp_path):
    assert unfinished_recordings(tmp_path / "never-created", tmp_path) == []

def test_dropped_fraction_limit_passes_the_recordings_that_were_always_fine():
    # Every recording from 27 Jul to 3 Aug captured 85-87% of real time and
    # transcribed correctly, so a standing ~15% loss must not warn. The two
    # meetings that came back as hallucination loops captured 15% and 20%. Those
    # files measure 0% here because they predate the filler that leaves the
    # evidence; a throttled capture recorded with the current command measured
    # 88.8% dropped on this machine, 2026-08-05.
    assert 0.20 < DROPPED_FRACTION_LIMIT < 0.60
