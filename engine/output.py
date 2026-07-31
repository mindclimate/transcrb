import json
import logging
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from engine.types import TranscriptResult

log = logging.getLogger(__name__)

# Transcribing a recording a second time — a different speaker count, a language
# picked by hand — used to overwrite the first result where it stood, so the two
# could not be compared and nothing said the earlier one had gone. The previous
# run moves in here instead. The canonical names still hold the newest result,
# so anything reading transcript.md does not have to know this exists.
ARCHIVE_DIRNAME = "previous"

# Named explicitly rather than globbed: whatever else ends up in the folder, the
# archive directory itself must never be swept into a later archive.
_ARTEFACTS = ("transcript.md", "transcript.srt", "transcript.json", "words.json")

def _archive_previous_run(dest: Path) -> Path | None:
    """Move an earlier run's files aside. Returns where they went, or None."""
    existing = [dest / n for n in _ARTEFACTS if (dest / n).is_file()]
    if not existing:
        return None
    # Labelled by when that transcript was produced, not by now, so the folder
    # name means something when you come back to it.
    made = min(p.stat().st_mtime for p in existing)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(made))
    archive = dest / ARCHIVE_DIRNAME / stamp
    n = 2
    while archive.exists():        # two runs can land in the same second
        archive = dest / ARCHIVE_DIRNAME / f"{stamp}-{n}"
        n += 1
    archive.mkdir(parents=True)
    for path in existing:
        path.rename(archive / path.name)
    log.info("Kept the previous transcript for %s in %s", dest.name, archive)
    return archive

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
    _archive_previous_run(dest)
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
