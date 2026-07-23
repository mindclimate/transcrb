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
