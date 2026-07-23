import re
import subprocess
from pathlib import Path

def build_record_command(device: str, out: Path, platform: str) -> list[str]:
    fmt = "avfoundation" if platform == "darwin" else "dshow"
    return [
        "ffmpeg", "-y", "-f", fmt, "-i", device,
        "-ac", "1", "-ar", "16000", str(out),
    ]

def list_input_devices(platform: str) -> list[str]:
    fmt = "avfoundation" if platform == "darwin" else "dshow"
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", fmt, "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    )
    # device lines look like: [AVFoundation ...] [0] MacBook Pro Microphone
    devices = []
    for line in proc.stderr.splitlines():
        m = re.search(r"\]\s*\[\d+\]\s*(.+)$", line)
        if m:
            devices.append(m.group(1).strip())
    return devices

def start_recording(device: str, out: Path, platform: str) -> subprocess.Popen:
    cmd = build_record_command(device, Path(out), platform)
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def stop_recording(proc: subprocess.Popen) -> None:
    try:
        proc.communicate(input=b"q", timeout=10)
    except Exception:
        proc.terminate()
