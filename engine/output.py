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
    # In a mixed transcript the language of each line is worth stating: a reader
    # skimming Hebrew and English together cannot otherwise tell whether a line
    # was spoken in that language or rendered into it.
    mixed = len({seg.lang for seg in result.segments if seg.lang}) > 1
    for seg in result.segments:
        who = f"{seg.speaker} ({seg.lang})" if mixed and seg.lang else seg.speaker
        lines.append(f"[{fmt_timestamp(seg.start)}] {who}: {seg.text}")
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
    """The transcript as segments — the shape downstream ingestion reads.

    Word timings are deliberately excluded; they are bulky and live in
    words.json for re-segmentation.
    """
    data = asdict(result)
    data.pop("words", None)
    return json.dumps(data, ensure_ascii=False, indent=2)

def to_words_json(result: TranscriptResult) -> str:
    return json.dumps(
        {"language": result.language, "model": result.model,
         "duration": result.duration,
         "words": [asdict(w) for w in result.words]},
        ensure_ascii=False,
    )

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
    if result.words:
        paths["words"] = dest / "words.json"
        paths["words"].write_text(to_words_json(result), encoding="utf-8")
    if inbox is not None:
        inbox = Path(inbox)
        inbox.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths["md"], inbox / f"{name}.md")
        shutil.copyfile(paths["json"], inbox / f"{name}.json")
    return paths
