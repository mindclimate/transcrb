"""The repository root is what someone sees when they unzip this and install it.

Only the files a person is meant to double-click or read belong there. Everything
else — app code, config, setup scripts, docs — lives in a subfolder, so a
non-technical friend is not asked to guess which of fifteen files to open.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# pyproject.toml has to stay: `uv pip install -e .` reads it from the root, and
# setuptools resolves the engine/ and web/ packages relative to it.
ALLOWED_ROOT_FILES = {
    "README.md",
    "First-time setup.command",
    "First-time setup.bat",
    "Start Transcrb.command",
    "Start Transcrb.bat",
    "pyproject.toml",
}

def _visible_root_files() -> set[str]:
    return {p.name for p in ROOT.iterdir()
            if p.is_file() and not p.name.startswith(".")}

def test_root_holds_only_readme_install_and_run_files():
    assert _visible_root_files() == ALLOWED_ROOT_FILES

def test_the_launchers_a_friend_double_clicks_are_present_and_executable():
    for name in ("First-time setup.command", "Start Transcrb.command"):
        launcher = ROOT / name
        assert launcher.is_file(), f"{name} is missing from the root"
        assert launcher.stat().st_mode & 0o111, f"{name} is not executable"

def test_setup_and_config_moved_into_subfolders():
    assert (ROOT / "scripts" / "setup.sh").is_file()
    assert (ROOT / "config" / "config.example.toml").is_file()
    assert not (ROOT / "setup.sh").exists()
    assert not (ROOT / "cli.py").exists()
