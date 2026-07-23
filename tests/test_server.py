from pathlib import Path
from fastapi.testclient import TestClient
from engine.types import TranscriptResult, Segment
from web.server import create_app

def _runner(src, cfg, lang, diarize):
    return TranscriptResult(language="en", model="large-v3", duration=1.0,
                            segments=[Segment(0.0, 1.0, "SPEAKER_00", "hi")])

def _client(tmp_path):
    from engine.config import Config
    cfg = Config(output_dir=tmp_path / "out", inbox=None, hf_token=None,
                 compute_type="int8", fallback_language="en")
    return TestClient(create_app(cfg=cfg, runner=_runner))

def test_health(tmp_path):
    r = _client(tmp_path).get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

def test_transcribe_endpoint(tmp_path):
    client = _client(tmp_path)
    files = {"file": ("call.wav", b"fakebytes", "audio/wav")}
    r = client.post("/api/transcribe", files=files, data={"lang": "en", "diarize": "false"})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "call"
    assert Path(body["md"]).exists()
