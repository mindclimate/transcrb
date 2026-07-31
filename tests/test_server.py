from pathlib import Path
from fastapi.testclient import TestClient
from engine.types import TranscriptResult, Segment
from web.server import create_app

def _runner(src, cfg, lang, diarize, progress=None, num_speakers=None):
    if progress:
        progress('transcribing')
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

def test_file_endpoint_rejects_outside_output_dir(tmp_path):
    client = _client(tmp_path)
    r = client.get("/api/file", params={"path": "/etc/passwd"})
    assert r.status_code == 403

def test_file_endpoint_404_for_missing(tmp_path):
    client = _client(tmp_path)
    missing = tmp_path / "out" / "nope" / "transcript.md"
    r = client.get("/api/file", params={"path": str(missing)})
    assert r.status_code == 404

def test_transcribe_sanitizes_path_traversal_filename(tmp_path):
    client = _client(tmp_path)
    files = {"file": ("../../evil.wav", b"x", "audio/wav")}
    r = client.post("/api/transcribe", files=files, data={"lang": "en", "diarize": "false"})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "evil"
    expected_md = tmp_path / "out" / "evil" / "transcript.md"
    assert Path(body["md"]) == expected_md
    assert expected_md.exists()
    # ensure nothing was written outside the configured output_dir
    assert (tmp_path / "out").resolve() in expected_md.resolve().parents

def test_favicon_returns_no_content(tmp_path):
    assert _client(tmp_path).get("/favicon.ico").status_code == 204

def test_progress_reports_stage_after_transcribe(tmp_path):
    client = _client(tmp_path)
    files = {"file": ("call.wav", b"x", "audio/wav")}
    client.post("/api/transcribe", files=files, data={"job": "job-1"})
    assert client.get("/api/progress", params={"job": "job-1"}).json()["stage"] == "done"

def test_progress_unknown_job_is_blank(tmp_path):
    assert _client(tmp_path).get("/api/progress", params={"job": "nope"}).json()["stage"] == ""

def test_transcribe_error_returns_500_json_not_traceback(tmp_path):
    from engine.config import Config
    from web.server import create_app
    from fastapi.testclient import TestClient
    def boom(*a, **k): raise RuntimeError("model exploded")
    cfg = Config(output_dir=tmp_path / "out", inbox=None, hf_token=None,
                 compute_type="int8", fallback_language="en")
    c = TestClient(create_app(cfg=cfg, runner=boom), raise_server_exceptions=False)
    r = c.post("/api/transcribe", files={"file": ("a.wav", b"x", "audio/wav")})
    assert r.status_code == 500
    assert "model exploded" in r.json()["error"]

def test_transcribe_without_file_returns_400(tmp_path):
    r = _client(tmp_path).post("/api/transcribe", data={"lang": "en"})
    assert r.status_code == 400

def test_server_path_outside_output_dir_rejected(tmp_path):
    r = _client(tmp_path).post("/api/transcribe", data={"server_path": "/etc/passwd"})
    assert r.status_code == 400

def test_devices_endpoint_returns_list(tmp_path):
    r = _client(tmp_path).get("/api/devices")
    assert r.status_code == 200 and isinstance(r.json()["devices"], list)

def test_record_stop_unknown_id_returns_404(tmp_path):
    r = _client(tmp_path).post("/api/record/stop", data={"rec_id": "nope"})
    assert r.status_code == 404


# ---- silent-input guards -----------------------------------------------------
# A silent device produces a large, valid, useless file. Size alone cannot tell
# the difference, so the record flow checks levels before, during, and after.

import numpy as np
import soundfile as sf
import web.server as S
from engine.record import SILENCE_FLOOR_DBFS

def _client_with_probe(tmp_path, peak):
    from engine.config import Config
    cfg = Config(output_dir=tmp_path / "out", inbox=None, hf_token=None,
                 compute_type="int8", fallback_language="en")
    return TestClient(create_app(cfg=cfg, runner=_runner,
                                 probe=lambda device, platform: peak))

def _fake_recorder(monkeypatch, samples):
    """Stand in for ffmpeg: writes the wav a real recording would leave behind."""
    def _start(device, out, platform):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), np.asarray(samples, dtype="float32"), 16000, subtype="PCM_16")
        return object()
    monkeypatch.setattr(S.recorder, "start_recording", _start)
    monkeypatch.setattr(S.recorder, "stop_recording", lambda proc: None)

def test_check_flags_a_silent_device_with_the_multi_output_fix(tmp_path):
    client = _client_with_probe(tmp_path, SILENCE_FLOOR_DBFS)
    r = client.post("/api/record/check", data={"device": ":1", "name": "BlackHole 2ch"})
    assert r.status_code == 200
    body = r.json()
    assert body["silent"] is True
    assert "Multi-Output" in body["hint"]

def test_check_passes_a_live_device(tmp_path):
    client = _client_with_probe(tmp_path, -32.0)
    body = client.post("/api/record/check",
                       data={"device": ":2", "name": "MacBook Pro Microphone"}).json()
    assert body["silent"] is False
    assert body["peak_dbfs"] == -32.0

def test_check_reports_a_device_that_cannot_be_opened(tmp_path):
    from engine.config import Config
    cfg = Config(output_dir=tmp_path / "out", inbox=None, hf_token=None,
                 compute_type="int8", fallback_language="en")
    def _boom(device, platform):
        raise RuntimeError("could not open the input device: busy")
    client = TestClient(create_app(cfg=cfg, runner=_runner, probe=_boom))
    body = client.post("/api/record/check", data={"device": ":9", "name": "Gone"}).json()
    assert "could not open" in body["error"]

def test_stop_flags_a_silent_recording_instead_of_transcribing_it(tmp_path, monkeypatch):
    _fake_recorder(monkeypatch, np.zeros(16000 * 2))
    client = _client_with_probe(tmp_path, -30.0)
    rec_id = client.post("/api/record/start",
                         data={"device": ":1", "name": "BlackHole 2ch"}).json()["rec_id"]
    body = client.post("/api/record/stop", data={"rec_id": rec_id}).json()
    assert body["silent"] is True
    assert "Multi-Output" in body["hint"]      # names the device chosen at start
    assert body["bytes"] > 0                   # the file is kept, not deleted

def test_stop_accepts_a_recording_with_audio_in_it(tmp_path, monkeypatch):
    t = np.linspace(0, 2, 32000, endpoint=False)
    _fake_recorder(monkeypatch, 0.5 * np.sin(2 * np.pi * 440 * t))
    client = _client_with_probe(tmp_path, -30.0)
    rec_id = client.post("/api/record/start",
                         data={"device": ":2", "name": "MacBook Pro Microphone"}).json()["rec_id"]
    body = client.post("/api/record/stop", data={"rec_id": rec_id}).json()
    assert body["silent"] is False
    assert "hint" not in body

def test_level_reports_no_signal_while_recording(tmp_path, monkeypatch):
    _fake_recorder(monkeypatch, np.zeros(16000 * 3))
    client = _client_with_probe(tmp_path, -30.0)
    rec_id = client.post("/api/record/start",
                         data={"device": ":1", "name": "BlackHole 2ch"}).json()["rec_id"]
    body = client.get("/api/record/level", params={"rec_id": rec_id}).json()
    assert body["silent"] is True
    assert body["peak_dbfs"] == SILENCE_FLOOR_DBFS

def test_level_for_unknown_recording_is_404(tmp_path):
    r = _client_with_probe(tmp_path, -30.0).get("/api/record/level",
                                                params={"rec_id": "nope"})
    assert r.status_code == 404


# ---- automatic system-audio capture ------------------------------------------
# On macOS 14.2+ a CoreAudio tap captures the call wherever it is playing, so
# there is nothing to route and no output device to switch. The Record panel
# offers it as one entry and builds the device on demand.

from web.server import AUTO_DEVICE_ID

class _FakeCapture:
    """Stands in for the tap-backed device without touching CoreAudio."""
    instances = []

    def __init__(self):
        self.started = self.stopped = False
        _FakeCapture.instances.append(self)

    def start(self):
        self.started = True
        return ":7"

    def stop(self):
        self.stopped = True

def _capture_client(tmp_path, capture=_FakeCapture, peak=-30.0):
    from engine.config import Config
    cfg = Config(output_dir=tmp_path / "out", inbox=None, hf_token=None,
                 compute_type="int8", fallback_language="en")
    return TestClient(create_app(cfg=cfg, runner=_runner, capture=capture,
                                 probe=lambda device, platform: peak))

def test_devices_offers_automatic_capture_first(tmp_path):
    body = _capture_client(tmp_path).get("/api/devices").json()
    assert body["devices"][0]["id"] == AUTO_DEVICE_ID
    assert "automatic" in body["devices"][0]["name"].lower()

def test_devices_omits_automatic_capture_when_unsupported(tmp_path):
    body = _capture_client(tmp_path, capture=False).get("/api/devices").json()
    assert all(d["id"] != AUTO_DEVICE_ID for d in body["devices"])

def test_start_builds_the_capture_device_and_records_from_it(tmp_path, monkeypatch):
    _FakeCapture.instances = []
    used = {}
    def _start(device, out, platform):
        used["device"] = device
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), np.zeros(1600, dtype="float32"), 16000, subtype="PCM_16")
        return object()
    monkeypatch.setattr(S.recorder, "start_recording", _start)
    monkeypatch.setattr(S.recorder, "stop_recording", lambda proc: None)
    client = _capture_client(tmp_path)
    r = client.post("/api/record/start", data={"device": AUTO_DEVICE_ID, "name": "auto"})
    assert r.status_code == 200
    assert used["device"] == ":7"          # the id the capture device reported
    assert _FakeCapture.instances[-1].started

def test_stop_tears_down_the_capture_device(tmp_path, monkeypatch):
    _FakeCapture.instances = []
    def _start(device, out, platform):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), np.zeros(1600, dtype="float32"), 16000, subtype="PCM_16")
        return object()
    monkeypatch.setattr(S.recorder, "start_recording", _start)
    monkeypatch.setattr(S.recorder, "stop_recording", lambda proc: None)
    client = _capture_client(tmp_path)
    rec_id = client.post("/api/record/start",
                         data={"device": AUTO_DEVICE_ID}).json()["rec_id"]
    client.post("/api/record/stop", data={"rec_id": rec_id})
    # The tap belongs to this process; leaving it alive leaks a device that
    # then shows up in every other app's input list.
    assert _FakeCapture.instances[-1].stopped

def test_start_reports_a_capture_device_that_cannot_be_built(tmp_path):
    class _Broken:
        def start(self): raise RuntimeError("CoreAudio refused the system-audio tap")
        def stop(self): pass
    client = _capture_client(tmp_path, capture=_Broken)
    r = client.post("/api/record/start", data={"device": AUTO_DEVICE_ID})
    assert r.status_code == 500
    assert "tap" in r.json()["error"]

def test_check_probes_the_capture_device_and_releases_it(tmp_path):
    _FakeCapture.instances = []
    body = _capture_client(tmp_path).post("/api/record/check",
                                          data={"device": AUTO_DEVICE_ID}).json()
    assert body["silent"] is False
    assert _FakeCapture.instances[-1].stopped    # not left holding the device

def test_stop_releases_the_tap_even_when_stopping_the_recorder_fails(tmp_path, monkeypatch):
    """A tap outlives the process that forgot it, polluting every app's input list.

    Releasing it must not depend on the recorder shutting down cleanly.
    """
    _FakeCapture.instances = []
    def _start(device, out, platform):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), np.zeros(1600, dtype="float32"), 16000, subtype="PCM_16")
        return object()
    monkeypatch.setattr(S.recorder, "start_recording", _start)
    monkeypatch.setattr(S.recorder, "stop_recording",
                        lambda proc: (_ for _ in ()).throw(RuntimeError("ffmpeg would not die")))
    client = _capture_client(tmp_path)
    rec_id = client.post("/api/record/start",
                         data={"device": AUTO_DEVICE_ID}).json()["rec_id"]
    r = client.post("/api/record/stop", data={"rec_id": rec_id})
    assert r.status_code == 500
    assert "ffmpeg would not die" in r.json()["error"]
    assert _FakeCapture.instances[-1].stopped

def test_shutdown_releases_a_tap_left_open_by_an_unstopped_recording(tmp_path, monkeypatch):
    """Closing the terminal is how this app is stopped, mid-recording included."""
    _FakeCapture.instances = []
    killed = []
    def _start(device, out, platform):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), np.zeros(1600, dtype="float32"), 16000, subtype="PCM_16")
        return object()
    monkeypatch.setattr(S.recorder, "start_recording", _start)
    monkeypatch.setattr(S.recorder, "stop_recording", lambda proc: killed.append(proc))
    from engine.config import Config
    cfg = Config(output_dir=tmp_path / "out", inbox=None, hf_token=None,
                 compute_type="int8", fallback_language="en")
    app = create_app(cfg=cfg, runner=_runner, capture=_FakeCapture,
                     probe=lambda device, platform: -30.0)
    with TestClient(app) as client:            # context manager fires shutdown
        client.post("/api/record/start", data={"device": AUTO_DEVICE_ID})
    assert _FakeCapture.instances[-1].stopped
    assert killed, "the recorder process was left running"


# ---- one transcription at a time ---------------------------------------------
# Two at once on one machine is slower than one after the other: each loads its
# own multi-GB model, holds the whole recording in memory, and reaches for the
# same GPU. Nothing in the UI prevented a second click.

import threading
import time as _time

def _slot_client(tmp_path, runner):
    from engine.config import Config
    cfg = Config(output_dir=tmp_path / "out", inbox=None, hf_token=None,
                 compute_type="int8", fallback_language="en")
    return create_app(cfg=cfg, runner=runner)

def test_two_transcriptions_never_run_at_the_same_time(tmp_path):
    seen_together = []
    active = []
    guard = threading.Lock()

    def slow(src, cfg, lang, diarize, progress=None, num_speakers=None):
        with guard:
            active.append(1)
            if len(active) > 1:
                seen_together.append(True)
        _time.sleep(0.4)
        with guard:
            active.pop()
        return TranscriptResult(language="en", model="m", duration=1.0,
                                segments=[Segment(0.0, 1.0, "SPEAKER_00", "hi")])

    app = _slot_client(tmp_path, slow)
    def fire(n):
        TestClient(app).post("/api/transcribe",
                             files={"file": (f"c{n}.wav", b"x", "audio/wav")},
                             data={"diarize": "false"})
    threads = [threading.Thread(target=fire, args=(i,)) for i in range(2)]
    for t in threads: t.start()
    for t in threads: t.join(timeout=20)
    assert not seen_together, "two transcriptions overlapped"

def test_a_waiting_transcription_reports_that_it_is_queued(tmp_path):
    running = threading.Event()
    release = threading.Event()

    def blocking(src, cfg, lang, diarize, progress=None, num_speakers=None):
        running.set()
        release.wait(timeout=20)
        return TranscriptResult(language="en", model="m", duration=1.0,
                                segments=[Segment(0.0, 1.0, "SPEAKER_00", "hi")])

    app = _slot_client(tmp_path, blocking)
    def fire(job):
        TestClient(app).post("/api/transcribe",
                             files={"file": (f"{job}.wav", b"x", "audio/wav")},
                             data={"diarize": "false", "job": job})
    first = threading.Thread(target=fire, args=("job-a",))
    first.start()
    assert running.wait(timeout=10), "the first transcription never started"
    second = threading.Thread(target=fire, args=("job-b",))
    second.start()
    try:
        reader = TestClient(app)
        deadline = _time.time() + 10
        stage = ""
        while _time.time() < deadline:
            stage = reader.get("/api/progress", params={"job": "job-b"}).json()["stage"]
            if stage == "queued":
                break
            _time.sleep(0.05)
        assert stage == "queued", f"second job reported {stage!r}, not 'queued'"
    finally:
        release.set()
        first.join(timeout=20)
        second.join(timeout=20)
