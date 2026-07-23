import re
import subprocess
from pathlib import Path

def build_record_command(device: str, out: Path, platform: str) -> list[str]:
    fmt = "avfoundation" if platform == "darwin" else "dshow"
    return [
        "ffmpeg", "-y", "-f", fmt, "-i", device,
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
