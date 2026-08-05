import json
import logging
import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Anything quieter than this is treated as "no signal". A live mic in a quiet
# room still sits around -40 dBFS; digital silence measures -90 dBFS or below,
# so the gap is wide and the threshold does not need to be clever.
SILENCE_DBFS = -60.0
SILENCE_FLOOR_DBFS = -120.0

PROBE_SECONDS = 1.5

def build_record_command(device: str, out: Path, platform: str) -> list[str]:
    fmt = "avfoundation" if platform == "darwin" else "dshow"
    return [
        "ffmpeg", "-y", "-f", fmt, "-i", device,
        # Without this ffmpeg holds the whole recording in its muxer buffer and
        # the file stays 0 bytes until it exits, so nothing can read the levels
        # while the meeting is still running.
        "-flush_packets", "1",
        # Write against the input's timestamps rather than just concatenating
        # whatever arrived. ffmpeg's avfoundation input holds one pending audio
        # buffer and blocks the capture callback until it is read, and CoreAudio
        # cannot wait, so a busy machine loses whatever was produced during the
        # stall — no error, no gap, and the surviving samples, packed end to
        # end, replay the meeting at several times speed.
        # This turns the loss back into silence at the point it happened, which
        # keeps every later timestamp honest and keeps the speech intelligible.
        # first_pts=0 is not optional alongside it: avfoundation timestamps
        # start at the host uptime, so without it the stream begins hours in.
        "-af", "aresample=async=1:first_pts=0",
        "-ac", "1", "-ar", "16000", str(out),
    ]

def parse_device_listing(stderr: str, platform: str) -> list[dict]:
    """Audio input devices from ffmpeg's -list_devices output.

    The two backends print different shapes, and avfoundation lists video
    devices in the same numbered format as audio ones — so the audio section
    header has to be tracked rather than matching brackets alone.
    """
    devices: list[dict] = []
    if platform == "darwin":
        in_audio = False
        for line in stderr.splitlines():
            if "AVFoundation video devices" in line:
                in_audio = False
                continue
            if "AVFoundation audio devices" in line:
                in_audio = True
                continue
            if not in_audio:
                continue
            m = re.search(r"\[(\d+)\]\s*(.+?)\s*$", line)
            if m:
                # avfoundation addresses audio-only input as ":<index>"
                devices.append({"id": f":{m.group(1)}", "name": m.group(2)})
    else:
        for line in stderr.splitlines():
            if "(audio)" not in line:
                continue
            m = re.search(r'"([^"]+)"', line)
            if m:
                devices.append({"id": f"audio={m.group(1)}", "name": m.group(1)})
    return devices

def list_input_devices(platform: str) -> list[dict]:
    fmt = "avfoundation" if platform == "darwin" else "dshow"
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", fmt, "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    )
    return parse_device_listing(proc.stderr, platform)

# ---- the native recorder -----------------------------------------------------
# ffmpeg is the fallback, not the plan. Its avfoundation input holds one pending
# audio buffer and blocks the capture callback until it is read, which gives the
# whole capture about 10ms of tolerance before CoreAudio starts discarding a
# meeting. native/capture.swift keeps thirty seconds of ring buffer and writes
# from its own thread, and it counts what it loses instead of leaving it to be
# inferred. Measured on this machine under `taskpolicy -b` — the background
# throttle that destroyed two recorded meetings — ffmpeg lost 88.7% of a 30s
# capture from the built-in mic and the native recorder lost 0.1%.

NATIVE_RECORDER = Path(__file__).resolve().parent.parent / "native" / "transcrb-capture"

# It fails fast or not at all: a device that has gone away, or a microphone
# permission that was refused, exits within milliseconds. Falling back to ffmpeg
# after that costs a fraction of a second of the meeting. Past this point the
# native path is never abandoned — it is the one that survives a busy machine.
NATIVE_START_GRACE_SECONDS = 0.6

def native_recorder(platform: str, binary: Path = NATIVE_RECORDER) -> Path | None:
    """The native capture binary, if this machine is a Mac that has built it."""
    if platform != "darwin":
        return None
    return Path(binary) if os.access(binary, os.X_OK) else None

def progress_path(out: Path) -> Path:
    return Path(out).with_suffix(".progress.json")

def build_native_command(device: str, out: Path, binary: Path,
                         progress: Path | None = None) -> list[str]:
    """Record `device` — an avfoundation ':N' — with the native recorder.

    The index is resolved by the recorder through the same AVCaptureDevice list
    ffmpeg walks, so ':2' means the same input to both. Resolving it against
    CoreAudio's device list instead would be a different list in a different
    order, and would silently record the wrong device.
    """
    cmd = [str(binary), "--device-index", str(device).lstrip(":"),
           "--out", str(out)]
    if progress is not None:
        cmd += ["--progress", str(progress)]
    return cmd

def parse_capture_summary(raw: bytes | str | None) -> dict:
    """The JSON line the native recorder prints when it finishes.

    ffmpeg prints no such thing, so an empty result means "this recorder does
    not know how much it lost", not "it lost nothing".
    """
    if not raw:
        return {}
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return {}

def read_progress(out: Path) -> dict:
    """How the recording is going, published once a second while it runs."""
    try:
        return json.loads(progress_path(out).read_text())
    except Exception:
        # Absent in the first second, and never worth taking a recording down.
        return {}

def reported_dropped(stats: dict, wav: Path) -> float:
    """How much of the meeting never reached the file, 0 to 1.

    The native recorder counts the frames CoreAudio never handed it, which is
    the true number. ffmpeg cannot, so a recording it made is measured by how
    much of the file is filled-in silence — a good proxy that also counts a
    genuinely silent room. Prefer the count that cannot be confused.
    """
    if isinstance(stats, dict) and "dropped" in stats:
        try:
            return max(0.0, min(1.0, float(stats["dropped"])))
        except (TypeError, ValueError):
            log.warning("The recorder reported an unusable loss figure: %r",
                        stats.get("dropped"))
    return dropped_fraction(wav)

def start_recording(device: str, out: Path, platform: str) -> subprocess.Popen:
    out = Path(out)
    binary = native_recorder(platform)
    if binary is not None:
        proc = subprocess.Popen(
            build_native_command(device, out, binary, progress_path(out)),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(NATIVE_START_GRACE_SECONDS)
        if proc.poll() is None:
            return proc
        why = (proc.stderr.read() if proc.stderr else b"") or b""
        log.warning("The native recorder would not start (%s); falling back to "
                    "ffmpeg, which loses audio on a busy machine",
                    why.decode("utf-8", "replace").strip() or "no reason given")
    cmd = build_record_command(device, out, platform)
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def recorder_name(proc) -> str:
    """Which recorder is running: 'native' or 'ffmpeg'. Their advice differs.

    Anything unrecognisable counts as ffmpeg, because the only thing this
    decides is which advice to print, and identifying a recorder must never be
    the reason a meeting fails to start.
    """
    args = getattr(proc, "args", None)
    if isinstance(args, (list, tuple)):
        args = args[0] if args else None
    if args is None:
        return "ffmpeg"
    return "native" if Path(str(args)).name == NATIVE_RECORDER.name else "ffmpeg"

def stop_recording(proc: subprocess.Popen) -> dict:
    """Ask the recorder to finish cleanly, then make sure it is actually reaped.

    Returns whatever it reported about the recording; empty for ffmpeg, which
    reports nothing. Both recorders stop on 'q' or on stdin closing.
    """
    raw = None
    try:
        raw, _ = proc.communicate(input=b"q", timeout=15)
    except Exception:
        proc.terminate()
        try:
            raw, _ = proc.communicate(timeout=5)
        except Exception:
            proc.kill()
            try:
                raw, _ = proc.communicate(timeout=5)
            except Exception:
                pass
    return parse_capture_summary(raw)

# ---- level metering ----------------------------------------------------------
# ffmpeg exits 0 and writes a perfectly valid file when a device hands it
# nothing but zeros, so "the recording exists" proves nothing. Every guard in
# the record flow is built on measuring the actual samples.

def _data_span(f) -> tuple[int, int]:
    """Byte offset and declared length of a wav's data chunk."""
    f.seek(0)
    head = f.read(12)
    if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos = 12
    while True:
        f.seek(pos)
        header = f.read(8)
        if len(header) < 8:
            raise ValueError("wav has no data chunk")
        chunk_id, size = header[:4], int.from_bytes(header[4:8], "little")
        if chunk_id == b"data":
            return pos + 8, size
        pos += 8 + size + (size & 1)   # chunks are word-aligned

def _read_samples(wav: Path, tail_seconds: float | None = None,
                  rate: int = 16000) -> np.ndarray:
    """The 16-bit samples of a wav, optionally only its last few seconds.

    Reads off disk rather than trusting the header: while ffmpeg is still
    recording, the data chunk's declared size is 0, and the LIST/INFO chunk it
    writes means the samples do not start at a fixed offset.
    """
    path = Path(wav)
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        start, declared = _data_span(f)
        end = start + declared if declared and start + declared <= file_size \
            else file_size
        end -= (end - start) % 2               # keep reads frame-aligned
        if tail_seconds is not None:
            start = max(start, end - int(tail_seconds * rate) * 2)
        if end <= start:
            return np.empty(0, dtype="<i2")
        f.seek(start)
        raw = f.read(end - start)
    return np.frombuffer(raw, dtype="<i2")

def peak_dbfs(wav: Path, tail_seconds: float | None = None,
              rate: int = 16000) -> float:
    """Peak level of a 16-bit PCM wav in dBFS, or the floor if it is silent."""
    samples = _read_samples(Path(wav), tail_seconds, rate)
    if samples.size == 0:
        return SILENCE_FLOOR_DBFS
    peak = int(np.abs(samples.astype(np.int32)).max())
    if peak == 0:
        return SILENCE_FLOOR_DBFS
    return max(SILENCE_FLOOR_DBFS, 20.0 * math.log10(peak / 32768.0))

def is_silent(dbfs: float) -> bool:
    return dbfs < SILENCE_DBFS

def silence_hint(device_name: str) -> str:
    """What to actually do about a dead input, named for the device chosen."""
    if "blackhole" in (device_name or "").lower():
        return (f"{device_name} received no audio. It is a virtual output: it only "
                "carries sound when macOS is playing into it. Open Audio MIDI Setup, "
                "create a Multi-Output Device containing both BlackHole 2ch and your "
                "headphones, then select that Multi-Output Device as the system output "
                "(Sound settings, or option-click the volume icon). Plugging or "
                "unplugging headphones resets the output, so re-check it before a call.")
    name = device_name or "the selected input"
    return (f"{name} received no audio. Check it is not muted, that its input volume "
            "is up in Sound settings, and that Transcrb is allowed to use the "
            "microphone in Privacy & Security → Microphone.")

# ---- dropped audio -----------------------------------------------------------
# A loud recording is not a complete one. When the capture layer drops buffers,
# the filler in the record command writes digital zeros in their place, and a
# run of those is the one direct measure of how much of the meeting never made
# it to disk. Ordinary audio crosses zero constantly but never sits there, so
# only runs long enough to be a lost buffer are counted.

# 13-15% was the standing loss on this machine while ffmpeg did the recording,
# and those recordings transcribed correctly; the two that came back as
# hallucination loops had lost 73% and 80%. The native recorder loses nothing
# worth measuring, so the limit is now generous rather than tight — it exists to
# catch a recording that is mostly missing, not to police a few percent.
DROPPED_FRACTION_LIMIT = 0.35
_MIN_GAP_SECONDS = 0.02

def dropped_fraction(wav: Path, tail_seconds: float | None = None,
                     rate: int = 16000) -> float:
    """How much of a recording is filled-in silence rather than captured audio."""
    samples = _read_samples(Path(wav), tail_seconds, rate)
    if samples.size == 0:
        return 0.0
    silent = samples == 0
    # Run-length encode the silent mask and keep only the long runs.
    edges = np.flatnonzero(np.diff(silent.astype(np.int8)))
    starts = np.concatenate((np.zeros(1, dtype=np.int64), edges + 1))
    lengths = np.diff(np.concatenate((starts, [silent.size]))).astype(np.int64)
    long_enough = (lengths >= int(_MIN_GAP_SECONDS * rate)) & silent[starts]
    return float(lengths[long_enough].sum() / silent.size)

def dropped_hint(fraction: float, recorder: str = "ffmpeg") -> str:
    """What to actually do about a recording that lost most of its audio."""
    lost = (f"{fraction * 100:.0f}% of this recording is silence the capture "
            "layer dropped rather than audio, so much of the meeting is not in "
            "the file. ")
    if recorder == "native":
        # The native recorder holds thirty seconds of ring buffer and is immune
        # to the throttle, so losing audio through it means something larger:
        # the machine stalled for half a minute, or the device went away.
        return lost + ("The native recorder lost it, and it holds thirty seconds "
                       "of buffer, so this is not the usual background throttle. "
                       "Check logs/transcrb.log for what the machine was doing, "
                       "and whether the input device changed mid-meeting.")
    return lost + ("ffmpeg recorded this, and its capture tolerates about 10ms "
                   "of delay. Build the native recorder — bash scripts/"
                   "build-native.sh — and check that the launchd agent still "
                   "sets ProcessType to Interactive "
                   "(~/Library/LaunchAgents/com.aliyoop.transcrb.plist), then "
                   "reinstall with ./install-service.sh.")

# ---- recordings a crash left behind ------------------------------------------
# Closing the terminal, a restart, or the service being reinstalled mid-meeting
# all end a recording without anyone stopping it. The audio is on disk — both
# recorders write as they go for exactly this reason — but nothing points at it,
# so it is lost in the way that matters. One 93MB recording sat unnoticed in
# out/recordings for two days.

def repair_wav_header(wav: Path) -> bool:
    """Patch the sizes of a recording that was cut off. Returns whether it changed.

    A recorder killed mid-meeting never writes them, so the file claims zero
    samples. Everything in this project reads the bytes on disk and is
    unbothered, but QuickTime, Finder's preview and libsndfile all believe the
    header and call an hour of meeting empty.
    """
    path = Path(wav)
    file_size = path.stat().st_size
    with open(path, "r+b") as f:
        try:
            start, declared = _data_span(f)
        except ValueError:
            return False
        actual = file_size - start
        actual -= actual % 2
        if actual <= 0 or declared == actual:
            return False
        f.seek(start - 4)
        f.write(actual.to_bytes(4, "little"))
        f.seek(4)
        f.write((file_size - 8).to_bytes(4, "little"))
    return True

def _audio_bytes(wav: Path) -> int:
    """How many bytes of samples a wav actually holds on disk.

    Not `size - 44`: ffmpeg writes a LIST/INFO chunk before the samples, so the
    header is not a fixed length, and an interrupted recording declares a data
    size of zero while holding an hour of audio.
    """
    size = wav.stat().st_size
    with open(wav, "rb") as f:
        try:
            start, declared = _data_span(f)
        except ValueError:
            return 0
    end = start + declared if declared and start + declared <= size else size
    return max(0, end - start) & ~1

def unfinished_recordings(recordings_dir: Path, output_dir: Path) -> list[dict]:
    """Recordings with no transcript beside them, newest first."""
    directory = Path(recordings_dir)
    if not directory.is_dir():
        return []
    found = []
    for wav in sorted(directory.glob("*.wav"), reverse=True):
        if (Path(output_dir) / wav.stem / "transcript.md").exists():
            continue
        try:
            audio = _audio_bytes(wav)
        except OSError:
            continue
        if audio == 0:
            continue
        found.append({"path": str(wav), "name": wav.stem,
                      "bytes": wav.stat().st_size,
                      # 16 kHz mono 16-bit is what both recorders write.
                      "seconds": round(audio / 32000, 1)})
    return found

def stalled_hint(stall_seconds: float, device_name: str = "") -> str:
    """What to say about an input that has stopped delivering audio entirely.

    Different from dropped audio and worth its own message: nothing is being
    lost in transit, the device has simply stopped, and the recording will stay
    frozen until the input comes back or the meeting is restarted.
    """
    name = device_name or "the input"
    return (f"{name} has stopped sending audio: nothing has arrived for "
            f"{stall_seconds:.0f} seconds, and none of that time is in the "
            "recording. This is usually a device that went away — headphones "
            "unplugged, a Bluetooth headset switching profile, or a call app "
            "taking the device. The recorder reconnects by itself when the "
            "device comes back; if it does not, stop and start the recording "
            "again rather than letting the meeting run on into nothing.")

def build_probe_command(device: str, out: Path, platform: str,
                        seconds: float = PROBE_SECONDS) -> list[str]:
    """Capture a short sample, through whichever recorder will do the meeting.

    Probing with ffmpeg while recording with the native tool would test the
    wrong thing: a microphone permission granted to one and refused to the other
    would pass the check and then record silence for an hour.
    """
    binary = native_recorder(platform)
    if binary is not None:
        return build_native_command(device, out, binary) + ["--seconds", f"{seconds:g}"]
    fmt = "avfoundation" if platform == "darwin" else "dshow"
    return [
        "ffmpeg", "-y", "-f", fmt, "-i", device,
        "-t", f"{seconds:g}", "-ac", "1", "-ar", "16000", str(out),
    ]

def probe_device_peak(device: str, platform: str,
                      seconds: float = PROBE_SECONDS) -> float:
    """Capture a short sample from a device and report its peak level.

    Raises if the device cannot be opened at all — a silent device and an
    unavailable one need different messages.
    """
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "probe.wav"
        proc = subprocess.run(build_probe_command(device, out, platform, seconds),
                              capture_output=True, text=True,
                              # Both recorders stop when stdin closes, and the
                              # native one would otherwise wait out the meeting
                              # on a terminal it inherited.
                              stdin=subprocess.DEVNULL,
                              timeout=max(30.0, seconds * 10))
        if not out.exists() or out.stat().st_size <= 44:
            tail = (proc.stderr or "").strip().splitlines()[-3:]
            raise RuntimeError("could not open the input device: " + " ".join(tail))
        return peak_dbfs(out)
