import math
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

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

def start_recording(device: str, out: Path, platform: str) -> subprocess.Popen:
    cmd = build_record_command(device, Path(out), platform)
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def stop_recording(proc: subprocess.Popen) -> None:
    """Ask ffmpeg to finish cleanly, then make sure it is actually reaped."""
    try:
        proc.communicate(input=b"q", timeout=10)
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)

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

def peak_dbfs(wav: Path, tail_seconds: float | None = None,
              rate: int = 16000) -> float:
    """Peak level of a 16-bit PCM wav in dBFS, or the floor if it is silent.

    Reads the samples off disk rather than trusting the header: while ffmpeg is
    still recording, the data chunk's declared size is 0, and the LIST/INFO
    chunk it writes means the samples do not start at a fixed offset.
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
            return SILENCE_FLOOR_DBFS
        f.seek(start)
        raw = f.read(end - start)

    samples = np.frombuffer(raw, dtype="<i2")
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

def build_probe_command(device: str, out: Path, platform: str,
                        seconds: float = PROBE_SECONDS) -> list[str]:
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
                              capture_output=True, text=True)
        if not out.exists() or out.stat().st_size == 0:
            tail = (proc.stderr or "").strip().splitlines()[-3:]
            raise RuntimeError("could not open the input device: " + " ".join(tail))
        return peak_dbfs(out)
