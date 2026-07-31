from pathlib import Path
from engine.config import load_config, Config

def test_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.compute_type == "int8"
    assert cfg.fallback_language == "en"
    assert cfg.inbox is None

def test_reads_file_values(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    p = tmp_path / "config.toml"
    p.write_text(
        'output_dir = "out"\n'
        'inbox = "/tmp/brain/inbox"\n'
        'hf_token = "abc"\n'
        'compute_type = "int8_float16"\n'
        'fallback_language = "he"\n'
    )
    cfg = load_config(p)
    assert cfg.inbox == Path("/tmp/brain/inbox")
    assert cfg.compute_type == "int8_float16"
    assert cfg.fallback_language == "he"
    assert cfg.hf_token == "abc"

def test_env_overrides_file_token(tmp_path, monkeypatch):
    p = tmp_path / "config.toml"
    p.write_text('hf_token = "file-token"\n')
    monkeypatch.setenv("HF_TOKEN", "env-token")
    cfg = load_config(p)
    assert cfg.hf_token == "env-token"

def test_default_config_path_sits_in_the_config_folder():
    # Anchored to the project root, not the working directory: the launchers cd
    # into the folder, but a friend running things by hand may not.
    from engine.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT
    assert DEFAULT_CONFIG_PATH == PROJECT_ROOT / "config" / "config.toml"
    assert (PROJECT_ROOT / "config" / "config.example.toml").is_file()

def test_load_config_uses_the_default_path_when_given_nothing():
    from engine.config import load_config
    cfg = load_config()          # must not raise when config.toml is absent
    assert cfg.compute_type in ("int8", "int8_float16")

def test_a_relative_output_dir_is_anchored_to_the_project_root(tmp_path, monkeypatch):
    """`output_dir = "out"` must mean the same folder wherever it was started.

    The server resolves this path to decide which files it is allowed to serve,
    so a working directory that differs from the project root would both write
    transcripts somewhere unexpected and shift that boundary.
    """
    from engine.config import PROJECT_ROOT
    p = tmp_path / "config.toml"
    p.write_text('output_dir = "out"\n')
    monkeypatch.chdir(tmp_path)
    cfg = load_config(p)
    assert cfg.output_dir == PROJECT_ROOT / "out"

def test_an_absolute_output_dir_is_left_alone(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(f'output_dir = "{tmp_path / "somewhere"}"\n')
    assert load_config(p).output_dir == tmp_path / "somewhere"
