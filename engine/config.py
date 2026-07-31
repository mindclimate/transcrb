import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

# Anchored to the project root so the config is found no matter where the
# process was started from — the launchers cd here, a person running things by
# hand might not.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.toml"

@dataclass
class Config:
    output_dir: Path
    inbox: Path | None
    hf_token: str | None
    compute_type: str
    fallback_language: str

def _anchored(value: str) -> Path:
    """Resolve a configured directory against the project root, not the cwd.

    The server compares resolved paths against `output_dir` to decide which
    files it will serve, so a relative value would move both the transcripts and
    that boundary depending on where the process happened to be started.
    """
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

def load_config(path: Path | None = None) -> Config:
    """Load config/config.toml. A missing file is fine — defaults apply."""
    data = {}
    path = DEFAULT_CONFIG_PATH if path is None else path
    if Path(path).exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)
    inbox = data.get("inbox")
    hf_token = os.environ.get("HF_TOKEN") or data.get("hf_token")
    return Config(
        output_dir=_anchored(data.get("output_dir", "out")),
        inbox=Path(inbox) if inbox else None,
        hf_token=hf_token,
        compute_type=data.get("compute_type", "int8"),
        fallback_language=data.get("fallback_language", "en"),
    )
