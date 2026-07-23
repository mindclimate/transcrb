import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    output_dir: Path
    inbox: Path | None
    hf_token: str | None
    compute_type: str
    fallback_language: str

def load_config(path: Path | None = None) -> Config:
    data = {}
    if path is not None and Path(path).exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)
    inbox = data.get("inbox")
    hf_token = os.environ.get("HF_TOKEN") or data.get("hf_token")
    return Config(
        output_dir=Path(data.get("output_dir", "out")),
        inbox=Path(inbox) if inbox else None,
        hf_token=hf_token,
        compute_type=data.get("compute_type", "int8"),
        fallback_language=data.get("fallback_language", "en"),
    )
