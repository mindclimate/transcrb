import json
import shutil
from dataclasses import asdict
from pathlib import Path
from engine.types import TranscriptResult

def fmt_timestamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

def _srt_time(seconds: float) -> str:
    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000))
    if ms == 1000:
        whole += 1
        ms = 0
    return f"{fmt_timestamp(whole)},{ms:03d}"

def to_markdown(result: TranscriptResult) -> str:
    lines = [
        "# Transcript",
        "",
        f"- Language: {result.language}",
        f"- Model: {result.model}",
        f"- Duration: {fmt_timestamp(result.duration)}",
        "",
    ]
    for seg in result.segments:
        lines.append(f"[{fmt_timestamp(seg.start)}] {seg.speaker}: {seg.text}")
    return "\n".join(lines) + "\n"

def to_srt(result: TranscriptResult) -> str:
    blocks = []
    for i, seg in enumerate(result.segments, start=1):
        blocks.append(
            f"{i}\n{_srt_time(seg.start)} --> {_srt_time(seg.end)}\n"
            f"{seg.speaker}: {seg.text}\n"
        )
    return "\n".join(blocks)

def to_json(result: TranscriptResult) -> str:
    return json.dumps(asdict(result), ensure_ascii=False, indent=2)

def write_outputs(result: TranscriptResult, out_dir: Path, name: str,
                  inbox: Path | None = None) -> dict:
    dest = Path(out_dir) / name
    dest.mkdir(parents=True, exist_ok=True)
    paths = {
        "md": dest / "transcript.md",
        "srt": dest / "transcript.srt",
        "json": dest / "transcript.json",
    }
    paths["md"].write_text(to_markdown(result), encoding="utf-8")
    paths["srt"].write_text(to_srt(result), encoding="utf-8")
    paths["json"].write_text(to_json(result), encoding="utf-8")
    if inbox is not None:
        inbox = Path(inbox)
        inbox.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths["md"], inbox / f"{name}.md")
        shutil.copyfile(paths["json"], inbox / f"{name}.json")
    return paths
