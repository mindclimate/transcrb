import json
import re
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
    assert body["dropped"] < 0.01
    assert "hint" not in body

def test_stop_warns_when_most_of_the_recording_was_dropped(tmp_path, monkeypatch):
    # The failure that cost two meetings: loud audio, so every silence guard
    # passes, but most of the buffers never reached ffmpeg. Only the proportion
    # of filled-in silence tells the difference.
    t = np.linspace(0, 2, 32000, endpoint=False)
    speech = 0.5 * np.sin(2 * np.pi * 440 * t)
    _fake_recorder(monkeypatch, np.concatenate([speech, np.zeros(16000 * 12)]))
    client = _client_with_probe(tmp_path, -30.0)
    rec_id = client.post("/api/record/start",
                         data={"device": ":2", "name": "MacBook Pro Microphone"}).json()["rec_id"]
    body = client.post("/api/record/stop", data={"rec_id": rec_id}).json()
    assert body["silent"] is False          # loud enough to pass the old guards
    assert body["dropped"] > 0.35
    assert "ProcessType" in body["hint"]    # the fix, not just "something is wrong"

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


# ---- the native recorder, and losing audio in the open -----------------------
# Two meetings were lost because nothing said anything until the transcript came
# back as one phrase on repeat. The native recorder counts what CoreAudio never
# handed it and publishes that once a second, so a recording in trouble is
# visible while the meeting is still running.

def _native_recorder(monkeypatch, samples, *, stats=None, progress=None):
    """Stand in for native/transcrb-capture, summary and all."""
    class _Proc:
        args = [str(S.recorder.NATIVE_RECORDER), "--device-index", "2"]
    def _start(device, out, platform):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), np.asarray(samples, dtype="float32"), 16000, subtype="PCM_16")
        if progress is not None:
            S.recorder.progress_path(Path(out)).write_text(json.dumps(progress))
        return _Proc()
    monkeypatch.setattr(S.recorder, "start_recording", _start)
    monkeypatch.setattr(S.recorder, "stop_recording", lambda proc: stats or {})

def test_start_reports_which_recorder_got_the_meeting(tmp_path, monkeypatch):
    # ffmpeg loses audio on a busy machine and the native recorder does not, so
    # which one ran is the single most useful fact about a recording.
    _native_recorder(monkeypatch, np.zeros(16000))
    client = _client_with_probe(tmp_path, -30.0)
    body = client.post("/api/record/start", data={"device": ":2", "name": "Mic"}).json()
    assert body["recorder"] == "native"

def test_the_recorders_own_count_beats_measuring_silence(tmp_path, monkeypatch):
    # A meeting with long genuine pauses reads as full of gaps to the silence
    # measure. The native recorder knows it lost nothing, and it is right.
    # The pauses are a noise floor, which is what a live input actually returns
    # in a quiet room; exact zeros mean no input at all, not a quiet one.
    rng = np.random.default_rng(0)
    room = rng.normal(0.0, 0.001, 16000 * 10)
    _native_recorder(monkeypatch, room, stats={"dropped": 0.0})
    client = _client_with_probe(tmp_path, -30.0)
    rec_id = client.post("/api/record/start",
                         data={"device": ":2", "name": "Mic"}).json()["rec_id"]
    body = client.post("/api/record/stop", data={"rec_id": rec_id}).json()
    assert body["dropped"] == 0.0
    assert "hint" not in body or "ProcessType" not in body["hint"]

def test_a_recording_of_digital_silence_is_never_reported_as_healthy(tmp_path,
                                                                     monkeypatch):
    # The whole failure of 2026-08-20 in one test. The recorder was pointed at a
    # device that opens, never errors, and returns zeros; it counted every frame
    # it was handed and called the recording 40% lost. Occasional click
    # artifacts at the buffer boundaries kept the peak at -5 dBFS, so the level
    # checks passed too. Three hours of meeting, and nothing said a word.
    dead = np.zeros(16000 * 120)
    dead[16000 * 30] = 0.5          # the one click that fooled every peak check
    _native_recorder(monkeypatch, dead, stats={"dropped": 0.40})
    client = _client_with_probe(tmp_path, -30.0)
    rec_id = client.post("/api/record/start",
                         data={"device": "com.transcrb.capture", "name": "Mic"}).json()["rec_id"]
    body = client.post("/api/record/stop", data={"rec_id": rec_id}).json()
    assert body["silent"] is True           # despite a peak of -6 dBFS
    assert body["dropped"] > 0.98           # despite the recorder claiming 0.40
    assert body["hint"]                     # and it says so rather than transcribing

def test_a_native_recording_that_lost_audio_is_not_blamed_on_the_throttle(tmp_path,
                                                                          monkeypatch):
    t = np.linspace(0, 2, 32000, endpoint=False)
    _native_recorder(monkeypatch, 0.5 * np.sin(2 * np.pi * 440 * t),
                     stats={"dropped": 0.61, "captured_seconds": 120.0})
    client = _client_with_probe(tmp_path, -30.0)
    rec_id = client.post("/api/record/start",
                         data={"device": ":2", "name": "Mic"}).json()["rec_id"]
    body = client.post("/api/record/stop", data={"rec_id": rec_id}).json()
    assert body["dropped"] == 0.61
    assert body["captured_seconds"] == 120.0
    assert "ProcessType" not in body["hint"]     # the native path is immune to it

def test_level_warns_about_lost_audio_while_the_meeting_is_still_running(tmp_path,
                                                                         monkeypatch):
    t = np.linspace(0, 2, 32000, endpoint=False)
    _native_recorder(monkeypatch, 0.5 * np.sin(2 * np.pi * 440 * t),
                     progress={"captured_seconds": 300.0, "dropped": 0.55,
                               "reason": "recording"})
    client = _client_with_probe(tmp_path, -30.0)
    rec_id = client.post("/api/record/start",
                         data={"device": ":2", "name": "Mic"}).json()["rec_id"]
    body = client.get("/api/record/level", params={"rec_id": rec_id}).json()
    assert body["silent"] is False
    assert body["dropped"] == 0.55
    assert body["captured_seconds"] == 300.0
    assert body["hint"]                          # said now, not an hour from now

def test_level_says_when_the_input_has_stopped_rather_than_slowed(tmp_path,
                                                                  monkeypatch):
    # A device that stops delivering is a different problem from one losing
    # buffers, and the thing to do about it is different. Without this the
    # counters simply freeze and the recording reads as healthy but short.
    t = np.linspace(0, 2, 32000, endpoint=False)
    _native_recorder(monkeypatch, 0.5 * np.sin(2 * np.pi * 440 * t),
                     progress={"captured_seconds": 300.0, "dropped": 0.12,
                               "stalled": True, "stall_seconds": 41.0,
                               "reason": "recording"})
    client = _client_with_probe(tmp_path, -30.0)
    rec_id = client.post("/api/record/start",
                         data={"device": ":2", "name": "AirPods"}).json()["rec_id"]
    body = client.get("/api/record/level", params={"rec_id": rec_id}).json()
    assert body["stalled"] is True
    assert "41 seconds" in body["hint"]
    assert "AirPods" in body["hint"]
    assert "ProcessType" not in body["hint"]   # not the throttle, a dead device

def test_level_says_nothing_about_loss_when_the_recording_is_healthy(tmp_path,
                                                                     monkeypatch):
    t = np.linspace(0, 2, 32000, endpoint=False)
    _native_recorder(monkeypatch, 0.5 * np.sin(2 * np.pi * 440 * t),
                     progress={"captured_seconds": 300.0, "dropped": 0.0,
                               "reason": "recording"})
    client = _client_with_probe(tmp_path, -30.0)
    rec_id = client.post("/api/record/start",
                         data={"device": ":2", "name": "Mic"}).json()["rec_id"]
    body = client.get("/api/record/level", params={"rec_id": rec_id}).json()
    assert body["dropped"] == 0.0
    assert "hint" not in body

def test_unfinished_recordings_are_offered_rather_than_left_on_disk(tmp_path):
    client = _client_with_probe(tmp_path, -30.0)
    recordings = tmp_path / "out" / "recordings"
    recordings.mkdir(parents=True)
    t = np.linspace(0, 2, 32000, endpoint=False)
    sf.write(str(recordings / "recording-20260803-120243.wav"),
             (0.5 * np.sin(2 * np.pi * 440 * t)).astype("float32"), 16000,
             subtype="PCM_16")
    body = client.get("/api/recordings/unfinished").json()
    assert [r["name"] for r in body["recordings"]] == ["recording-20260803-120243"]

def test_a_transcribed_recording_stops_being_offered(tmp_path):
    client = _client_with_probe(tmp_path, -30.0)
    recordings = tmp_path / "out" / "recordings"
    recordings.mkdir(parents=True)
    sf.write(str(recordings / "recording-20260803-120243.wav"),
             np.zeros(16000, dtype="float32"), 16000, subtype="PCM_16")
    done = tmp_path / "out" / "recording-20260803-120243"
    done.mkdir(parents=True)
    (done / "transcript.md").write_text("# Transcript")
    assert client.get("/api/recordings/unfinished").json()["recordings"] == []

def test_listing_repairs_a_recording_the_recorder_never_closed(tmp_path):
    # The service was reinstalled mid-meeting on 3 Aug. ffmpeg never wrote the
    # sizes, so libsndfile reads the file as empty even though the audio is all
    # there. Listing it is the moment to make it a real file again.
    import struct
    client = _client_with_probe(tmp_path, -30.0)
    recordings = tmp_path / "out" / "recordings"
    recordings.mkdir(parents=True)
    pcm = (np.asarray(np.sin(np.linspace(0, 200, 32000)) * 16000,
                      dtype="<i2")).tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
    body = (b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", 0) + pcm)
    cut = recordings / "recording-cut.wav"
    cut.write_bytes(b"RIFF" + struct.pack("<I", 0) + b"WAVE" + body)
    assert sf.info(str(cut)).frames == 0
    client.get("/api/recordings/unfinished")
    assert sf.info(str(cut)).frames == 32000

def test_stop_clears_the_progress_file_it_left_beside_the_recording(tmp_path,
                                                                    monkeypatch):
    _native_recorder(monkeypatch, np.zeros(16000),
                     progress={"dropped": 0.0, "reason": "recording"})
    client = _client_with_probe(tmp_path, -30.0)
    start = client.post("/api/record/start", data={"device": ":2", "name": "Mic"}).json()
    assert S.recorder.progress_path(Path(start["path"])).exists()
    client.post("/api/record/stop", data={"rec_id": start["rec_id"]})
    assert not S.recorder.progress_path(Path(start["path"])).exists()


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


# ---- one recorder at a time, and never a lost handle on it --------------------
# On 2026-08-20 the page was reloaded mid-meeting. The recording id lived in a
# JavaScript variable and nowhere else, so the reload disabled Stop and enabled
# Record; the next click built a second capture device, which destroyed the
# first one's device out from under it — same UID — and left two recorders
# fighting over one tap for the rest of the afternoon.

def _live_client(tmp_path, monkeypatch, samples=None):
    _native_recorder(monkeypatch, np.zeros(16000) if samples is None else samples,
                     stats={"dropped": 0.0})
    return _capture_client(tmp_path)

def test_a_second_record_click_never_starts_a_second_recorder(tmp_path, monkeypatch):
    client = _live_client(tmp_path, monkeypatch)
    first = client.post("/api/record/start", data={"device": AUTO_DEVICE_ID}).json()
    again = client.post("/api/record/start", data={"device": AUTO_DEVICE_ID})
    assert again.status_code == 409
    # It points at the running one rather than just refusing, so the page can
    # take hold of the meeting instead of leaving it unstoppable.
    assert again.json()["rec_id"] == first["rec_id"]

def test_checking_the_input_while_recording_cannot_destroy_the_device(tmp_path,
                                                                       monkeypatch):
    # The check builds a capture device to probe it. Building one destroys any
    # device already carrying the same UID, including the live meeting's.
    _FakeCapture.instances = []
    client = _live_client(tmp_path, monkeypatch)
    client.post("/api/record/start", data={"device": AUTO_DEVICE_ID})
    built = len(_FakeCapture.instances)
    r = client.post("/api/record/check", data={"device": AUTO_DEVICE_ID})
    assert r.status_code == 409
    assert len(_FakeCapture.instances) == built     # nothing new was built

def test_a_reloaded_page_can_find_the_recording_that_is_still_running(tmp_path,
                                                                       monkeypatch):
    client = _live_client(tmp_path, monkeypatch)
    started = client.post("/api/record/start",
                          data={"device": AUTO_DEVICE_ID, "title": "Priority sync"}).json()
    active = client.get("/api/record/active").json()["recording"]
    assert active["rec_id"] == started["rec_id"]
    assert active["title"] == "Priority sync"
    assert active["started_at"] > 0
    # And it is genuinely the same recording, not a description of one.
    assert client.post("/api/record/stop",
                       data={"rec_id": active["rec_id"]}).status_code == 200

def test_nothing_running_is_reported_as_nothing_rather_than_an_error(tmp_path):
    assert _capture_client(tmp_path).get("/api/record/active").json()["recording"] is None

def test_a_recording_still_being_written_is_not_offered_as_unfinished(tmp_path,
                                                                      monkeypatch):
    # Offering it invited transcribing half a meeting while the other half was
    # still being spoken, and rewriting its header under the recorder that owns it.
    client = _live_client(tmp_path, monkeypatch)
    started = client.post("/api/record/start", data={"device": AUTO_DEVICE_ID}).json()
    offered = client.get("/api/recordings/unfinished").json()["recordings"]
    assert all(r["path"] != started["path"] for r in offered)


# ---- naming a recording ------------------------------------------------------

STAMPED = re.compile(r"^\d{2}\.\d{2}\.\d{4}-\d{4}")

def test_a_named_recording_is_saved_under_that_name(tmp_path, monkeypatch):
    client = _live_client(tmp_path, monkeypatch)
    started = client.post("/api/record/start",
                          data={"device": AUTO_DEVICE_ID, "title": "RPO sync"}).json()
    name = Path(started["path"]).name
    assert STAMPED.match(name) and name.endswith(" RPO sync.wav")

def test_an_unnamed_recording_is_called_by_its_day_and_time(tmp_path, monkeypatch):
    client = _live_client(tmp_path, monkeypatch)
    started = client.post("/api/record/start", data={"device": AUTO_DEVICE_ID}).json()
    assert re.fullmatch(r"\d{2}\.\d{2}\.\d{4}-\d{4}\.wav", Path(started["path"]).name)

def test_a_name_cannot_write_outside_the_recordings_directory(tmp_path, monkeypatch):
    client = _live_client(tmp_path, monkeypatch)
    started = client.post("/api/record/start",
                          data={"device": AUTO_DEVICE_ID,
                                "title": "../../../etc/passwd"}).json()
    written = Path(started["path"]).resolve()
    assert (tmp_path / "out" / "recordings").resolve() == written.parent


# ---- naming a transcription of a dropped file --------------------------------

def test_a_named_upload_is_filed_under_that_name_not_the_filename(tmp_path):
    client = _client(tmp_path)
    r = client.post("/api/transcribe", files={"file": ("call.wav", b"x", "audio/wav")},
                    data={"lang": "en", "diarize": "false", "title": "RPO sync"})
    body = r.json()
    assert STAMPED.match(body["name"]) and body["name"].endswith(" RPO sync")
    assert Path(body["md"]).exists()

def test_an_unnamed_upload_is_still_filed_under_its_filename(tmp_path):
    # The file already has a name a person chose. Replacing it with a timestamp
    # would lose the only thing that says what it is.
    client = _client(tmp_path)
    r = client.post("/api/transcribe", files={"file": ("call.wav", b"x", "audio/wav")},
                    data={"lang": "en", "diarize": "false"})
    assert r.json()["name"] == "call"


# ---- renaming a finished transcript ------------------------------------------

def _transcribe(client, filename="call.wav", **data):
    return client.post("/api/transcribe",
                       files={"file": (filename, b"x", "audio/wav")},
                       data={"lang": "en", "diarize": "false", **data}).json()

def test_a_finished_transcript_can_be_renamed(tmp_path):
    client = _client(tmp_path)
    done = _transcribe(client)
    r = client.post("/api/rename", data={"name": done["name"], "title": "RPO sync"})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "RPO sync"
    assert (tmp_path / "out" / "RPO sync" / "transcript.md").exists()
    assert not (tmp_path / "out" / "call").exists()
    assert Path(body["md"]) == tmp_path / "out" / "RPO sync" / "transcript.md"

def test_renaming_a_recording_keeps_the_moment_it_was_made(tmp_path):
    client = _client(tmp_path)
    done = _transcribe(client, "02.09.2026-1912.wav")
    body = client.post("/api/rename",
                       data={"name": done["name"], "title": "RPO sync"}).json()
    assert body["name"] == "02.09.2026-1912 RPO sync"

def test_renaming_onto_an_existing_transcript_is_refused(tmp_path):
    # Renaming must never be the thing that loses a transcript.
    client = _client(tmp_path)
    _transcribe(client, "keep.wav")
    done = _transcribe(client, "call.wav")
    r = client.post("/api/rename", data={"name": done["name"], "title": "keep"})
    assert r.status_code == 409
    assert (tmp_path / "out" / "keep" / "transcript.md").exists()
    assert (tmp_path / "out" / "call" / "transcript.md").exists()

def test_renaming_something_that_is_not_there_is_a_404(tmp_path):
    client = _client(tmp_path)
    r = client.post("/api/rename", data={"name": "nothing", "title": "RPO sync"})
    assert r.status_code == 404

def test_a_rename_cannot_reach_outside_the_output_directory(tmp_path):
    client = _client(tmp_path)
    _transcribe(client)
    outside = tmp_path / "secret"
    outside.mkdir()
    r = client.post("/api/rename",
                    data={"name": "../secret", "title": "RPO sync"})
    assert r.status_code in (400, 404)
    assert outside.exists()

def test_a_rename_to_nothing_is_refused(tmp_path):
    client = _client(tmp_path)
    done = _transcribe(client)
    r = client.post("/api/rename", data={"name": done["name"], "title": "!!!"})
    assert r.status_code == 400
    assert (tmp_path / "out" / "call" / "transcript.md").exists()

def test_two_recordings_in_the_same_minute_do_not_overwrite_each_other(tmp_path, monkeypatch):
    # The stamp is only accurate to the minute, so a stopped-and-restarted
    # recording can ask for a name that is already taken.
    client = _live_client(tmp_path, monkeypatch)
    first = client.post("/api/record/start", data={"device": AUTO_DEVICE_ID}).json()
    client.post("/api/record/stop", data={"rec_id": first["rec_id"]})
    second = client.post("/api/record/start", data={"device": AUTO_DEVICE_ID}).json()
    assert second["path"] != first["path"]
    assert Path(first["path"]).exists()

def test_renaming_also_renames_the_copy_that_was_filed_elsewhere(tmp_path):
    # The inbox copy is what other tools see. Leaving it under the old name
    # makes the rename look like it did nothing.
    from engine.config import Config
    from fastapi.testclient import TestClient
    from web.server import create_app
    inbox = tmp_path / "inbox"
    cfg = Config(output_dir=tmp_path / "out", inbox=inbox, hf_token=None,
                 compute_type="int8", fallback_language="en")
    client = TestClient(create_app(cfg=cfg, runner=_runner))
    done = _transcribe(client)
    client.post("/api/rename", data={"name": done["name"], "title": "RPO sync"})
    assert (inbox / "RPO sync.md").exists()
    assert not (inbox / "call.md").exists()
