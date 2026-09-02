"""The brains a finished transcript can be filed into.

A brain is a brain-kit project: a folder with `brain/inbox/` in it, which is
where its ingestion picks new material up. They are found rather than listed in
config — they sit beside this project, and a hand-written list would go stale
every time one was added, renamed or cloned.
"""
import logging
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Where ingestion reads from, inside a brain-kit project.
INBOX_RELATIVE = Path("brain") / "inbox"

# Where the brain's own name is written, so the dropdown says "Work Brain"
# rather than "WorkBrain-page-currency".
BRAIN_TOML_RELATIVE = Path("brain") / "brain.toml"

@dataclass(frozen=True)
class Brain:
    slug: str          # the folder name — stable, and what the UI sends back
    name: str          # what to call it on screen
    inbox: Path

def _display_name(root: Path) -> str | None:
    """The name the brain gives itself, if it gives one."""
    toml = root / BRAIN_TOML_RELATIVE
    if not toml.is_file():
        return None
    try:
        with open(toml, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        # A brain being edited must not take the whole list down with it.
        log.warning("Could not read %s; using the folder name", toml)
        return None
    name = (data.get("brain") or {}).get("display_name")
    return str(name).strip() or None if name else None

def discover(root: Path | None) -> list[Brain]:
    """Every brain under `root`, by its folder name, in alphabetical order."""
    if root is None:
        return []
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for child in sorted(root.iterdir()):
        inbox = child / INBOX_RELATIVE
        if not inbox.is_dir():
            continue
        found.append(Brain(slug=child.name,
                           name=_display_name(child) or child.name,
                           inbox=inbox))
    return sorted(_disambiguated(found), key=lambda b: b.name.lower())

def _disambiguated(found: list[Brain]) -> list[Brain]:
    """Say which folder, where two brains call themselves the same thing.

    A brain cloned to work on it keeps the original's display_name, so four
    folders here answer to "Work Brain". A dropdown offering that four times is
    worse than no dropdown: filing into the wrong one is silent.
    """
    seen: dict[str, int] = {}
    for brain in found:
        seen[brain.name] = seen.get(brain.name, 0) + 1
    return [brain if seen[brain.name] == 1 or brain.name == brain.slug
            else Brain(brain.slug, f"{brain.name} ({brain.slug})", brain.inbox)
            for brain in found]

def find(root: Path | None, slug: str) -> Brain | None:
    for brain in discover(root):
        if brain.slug == slug:
            return brain
    return None

# What gets filed: the transcript a person reads and the structured form
# ingestion parses. Subtitles and word timings are of no use to a brain.
_FILED = (("transcript.md", ".md"), ("transcript.json", ".json"))

def file_transcript(transcript_dir: Path, name: str, brain: Brain) -> list[Path]:
    """Copy a finished transcript into a brain's inbox.

    Named after the transcript, not after the file inside it, because an inbox
    holds everything side by side and `transcript.md` says nothing there.
    Overwrites: the inbox is a drop point that gets consumed, so a second copy
    under a numbered name would only be ingested twice.
    """
    transcript_dir = Path(transcript_dir)
    brain.inbox.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, suffix in _FILED:
        src = transcript_dir / filename
        if not src.is_file():
            continue
        dest = brain.inbox / f"{name}{suffix}"
        shutil.copyfile(src, dest)
        written.append(dest)
    return written

def follow_rename(root: Path | None, old: str, new: str) -> list[Path]:
    """Rename copies already filed into any brain.

    A transcript is usually named properly only once it can be read, which is
    often after it has been filed. Leaving the old copy behind would have the
    brain ingest a meeting under a name that no longer exists anywhere else.
    """
    moved = []
    for brain in discover(root):
        for _filename, suffix in _FILED:
            filed = brain.inbox / f"{old}{suffix}"
            if filed.is_file():
                dest = brain.inbox / f"{new}{suffix}"
                filed.rename(dest)
                moved.append(dest)
    return moved
