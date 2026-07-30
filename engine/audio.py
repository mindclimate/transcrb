import shutil
import subprocess
from pathlib import Path
import soundfile as sf

def normalize_audio(src: Path, dst: Path) -> Path:
    src, dst = Path(src), Path(dst)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Run the first-time setup to install it.")
    if not src.exists():
        raise RuntimeError(f"Input audio not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
           "-vn", "-f", "wav", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg failed for {src.name}:\n{proc.stderr[-500:]}")
    return dst

def probe_duration(wav: Path) -> float:
    info = sf.info(str(wav))
    return info.frames / info.samplerate

def read_samples(wav: Path):
    """The whole normalized recording as float32 samples, plus its rate.

    Read once by the caller and sliced per span from memory. Reading inside the
    per-span loop instead costs a full file read for every span, which on a
    three-hour recording is most of a gigabyte each time round.
    """
    return sf.read(str(wav), dtype="float32")
