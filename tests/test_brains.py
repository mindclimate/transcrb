from pathlib import Path
from engine import brains

def _make_brain(root: Path, folder: str, display_name: str | None = None) -> Path:
    home = root / folder
    (home / "brain" / "inbox").mkdir(parents=True)
    if display_name:
        (home / "brain" / "brain.toml").write_text(
            f'[brain]\nslug = "{folder.lower()}"\n'
            f'display_name = "{display_name}"\n', encoding="utf-8")
    return home

def _transcript(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "transcript.md").write_text("# Transcript\n", encoding="utf-8")
    (d / "transcript.json").write_text('{"language": "en"}', encoding="utf-8")
    (d / "transcript.srt").write_text("1\n", encoding="utf-8")
    return d

def test_discovers_projects_that_have_an_inbox(tmp_path):
    _make_brain(tmp_path, "WorkBrain", "Work Brain")
    _make_brain(tmp_path, "Kadmi", "Kadmi — AI Education")
    (tmp_path / "NotABrain").mkdir()
    found = brains.discover(tmp_path)
    assert [b.slug for b in found] == ["Kadmi", "WorkBrain"]   # by display name
    assert [b.name for b in found] == ["Kadmi — AI Education", "Work Brain"]

def test_a_brain_without_a_toml_is_called_by_its_folder(tmp_path):
    _make_brain(tmp_path, "SideProject")
    assert [b.name for b in brains.discover(tmp_path)] == ["SideProject"]

def test_an_unreadable_toml_does_not_hide_the_brain(tmp_path):
    """A brain being edited must not vanish from the dropdown mid-keystroke."""
    home = _make_brain(tmp_path, "Broken")
    (home / "brain" / "brain.toml").write_text("[brain\nthis is not toml",
                                               encoding="utf-8")
    assert [b.name for b in brains.discover(tmp_path)] == ["Broken"]

def test_a_missing_root_is_not_an_error(tmp_path):
    assert brains.discover(tmp_path / "nowhere") == []
    assert brains.discover(None) == []

def test_find_by_slug(tmp_path):
    _make_brain(tmp_path, "WorkBrain", "Work Brain")
    assert brains.find(tmp_path, "WorkBrain").name == "Work Brain"
    assert brains.find(tmp_path, "Nope") is None

def test_files_the_transcript_under_its_own_name(tmp_path):
    """`transcript.md` says nothing in an inbox holding everything side by side."""
    home = _make_brain(tmp_path / "projects", "WorkBrain", "Work Brain")
    src = _transcript(tmp_path / "out", "02.09.2026-1200 Weekly with Vladi")
    brain = brains.find(tmp_path / "projects", "WorkBrain")
    written = brains.file_transcript(src, src.name, brain)
    inbox = home / "brain" / "inbox"
    assert (inbox / f"{src.name}.md").read_text() == "# Transcript\n"
    assert (inbox / f"{src.name}.json").is_file()
    # Subtitles are of no use to a brain and would only be ingested as noise.
    assert not (inbox / f"{src.name}.srt").exists()
    assert [p.name for p in written] == [f"{src.name}.md", f"{src.name}.json"]

def test_filing_twice_replaces_rather_than_duplicates(tmp_path):
    """The inbox is a drop point that gets consumed; a second numbered copy
    would be ingested as a second meeting."""
    home = _make_brain(tmp_path / "projects", "WorkBrain")
    src = _transcript(tmp_path / "out", "call")
    brain = brains.find(tmp_path / "projects", "WorkBrain")
    brains.file_transcript(src, "call", brain)
    (src / "transcript.md").write_text("# Transcript v2\n", encoding="utf-8")
    brains.file_transcript(src, "call", brain)
    inbox = home / "brain" / "inbox"
    assert [p.name for p in sorted(inbox.iterdir())] == ["call.json", "call.md"]
    assert (inbox / "call.md").read_text() == "# Transcript v2\n"

def test_rename_follows_the_copy_into_every_brain(tmp_path):
    """A meeting is usually named properly after it has been read — and filed."""
    a = _make_brain(tmp_path / "projects", "WorkBrain")
    b = _make_brain(tmp_path / "projects", "Kadmi")
    src = _transcript(tmp_path / "out", "call")
    for slug in ("WorkBrain", "Kadmi"):
        brains.file_transcript(src, "call", brains.find(tmp_path / "projects", slug))
    moved = brains.follow_rename(tmp_path / "projects", "call", "RPO sync")
    assert len(moved) == 4
    for home in (a, b):
        inbox = home / "brain" / "inbox"
        assert (inbox / "RPO sync.md").is_file()
        assert not (inbox / "call.md").exists()

def test_brains_that_share_a_name_say_which_folder_they_are(tmp_path):
    """A brain cloned to work on it keeps the original's display_name, so four
    folders can all answer to "Work Brain". Filing into the wrong one is
    silent, so the dropdown must not offer the same label twice."""
    _make_brain(tmp_path, "WorkBrain", "Work Brain")
    _make_brain(tmp_path, "WorkBrain-fork", "Work Brain")
    _make_brain(tmp_path, "Kadmi", "Kadmi")
    names = {b.slug: b.name for b in brains.discover(tmp_path)}
    assert names == {"WorkBrain": "Work Brain (WorkBrain)",
                     "WorkBrain-fork": "Work Brain (WorkBrain-fork)",
                     "Kadmi": "Kadmi"}
