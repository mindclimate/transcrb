import logging
import shutil
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.concurrency import run_in_threadpool
from engine.config import PROJECT_ROOT, load_config
from engine.logs import setup_logging
from engine.pipeline import transcribe_file
from engine.output import write_outputs
from engine import record as recorder
from engine.macos_audio import CAPTURE_DEVICE_NAME

log = logging.getLogger(__name__)
HERE = Path(__file__).parent
LOG_DIR = PROJECT_ROOT / "logs"

# Only one transcription runs at a time. Two at once on one machine is slower
# than one after the other — each loads its own multi-GB model, holds the whole
# recording in memory and reaches for the same GPU — and nothing in the UI
# stopped a second click. A waiting request reports "queued" rather than
# looking stalled.
_TRANSCRIBE_SLOT = threading.Semaphore(1)

# One entry in the input list stands for "build a system-audio tap on demand".
# It is not a real device until a recording starts.
AUTO_DEVICE_ID = "auto"
AUTO_DEVICE_NAME = "Call audio + my mic (automatic)"

def _default_capture_factory():
    """The tap-backed capture class, when this machine supports it."""
    if sys.platform != "darwin":
        return None
    try:
        from engine import macos_audio
    except Exception:
        return None
    return macos_audio.SystemCapture if macos_audio.taps_supported() else None

# Long transcriptions are opaque without this: the pipeline reports each stage
# against a client-supplied job id, and the page polls for it.
_PROGRESS: dict[str, dict] = {}
_PROGRESS_TTL_SECONDS = 3600

def _set_stage(job: str, stage: str) -> None:
    if job:
        _PROGRESS[job] = {"stage": stage, "at": time.time()}

def _expire_progress() -> None:
    cutoff = time.time() - _PROGRESS_TTL_SECONDS
    for k in [k for k, v in _PROGRESS.items() if v["at"] < cutoff]:
        _PROGRESS.pop(k, None)

def create_app(cfg=None, runner=None, probe=None, capture=None) -> FastAPI:
    """`capture` is the system-audio capture class: None auto-detects, False
    disables it, or pass a class for tests."""
    cfg = cfg or load_config()
    runner = runner or transcribe_file
    probe = probe or recorder.probe_device_peak
    capture_factory = _default_capture_factory() if capture is None else (capture or None)
    recordings: dict[str, dict] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        # Closing the terminal window is how this app is stopped, and that can
        # happen mid-recording. A tap outlives the process that forgot it and
        # then shows up in every other app's input list, so it is released here
        # even though the request that created it never came back.
        while recordings:
            _rec_id, entry = recordings.popitem()
            log.warning("Releasing a recording still running at shutdown: %s",
                        entry.get("path"))
            try:
                recorder.stop_recording(entry["proc"])
            except Exception:
                log.exception("Could not stop the recorder at shutdown")
            if entry.get("capture") is not None:
                try:
                    entry["capture"].stop()
                except Exception:
                    log.exception("Could not release the capture device at shutdown")

    app = FastAPI(lifespan=lifespan)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/")
    def index():
        return FileResponse(HERE / "index.html")

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    @app.get("/api/progress")
    def progress(job: str):
        _expire_progress()
        return _PROGRESS.get(job, {"stage": ""})

    async def _run_and_write(src: Path, name: str, lang: str, diarize: str,
                             speakers: str, job: str):
        lang_arg = lang or None
        n_spk = int(speakers) if speakers.strip().isdigit() and int(speakers) > 0 else None

        def _run_one_at_a_time():
            if not _TRANSCRIBE_SLOT.acquire(blocking=False):
                _set_stage(job, "queued")
                _TRANSCRIBE_SLOT.acquire()
            try:
                return runner(src, cfg, lang_arg, diarize.lower() != "false",
                              lambda stage: _set_stage(job, stage), n_spk)
            finally:
                _TRANSCRIBE_SLOT.release()

        result = await run_in_threadpool(_run_one_at_a_time)
        _set_stage(job, "writing")
        paths = write_outputs(result, cfg.output_dir, name, inbox=cfg.inbox)
        _set_stage(job, "done")
        return {"name": name, "md": str(paths["md"]), "json": str(paths["json"]),
                "speakers": sorted({s.speaker for s in result.segments}),
                "language": result.language}

    @app.post("/api/transcribe")
    async def transcribe(file: UploadFile | None = File(default=None),
                         server_path: str = Form(default=""),
                         lang: str = Form(default=""),
                         diarize: str = Form(default="true"),
                         speakers: str = Form(default=""),
                         job: str = Form(default="")):
        try:
            if server_path:
                # A file this server recorded — it never left the machine.
                src = Path(server_path).resolve()
                if cfg.output_dir.resolve() not in src.parents or not src.is_file():
                    return JSONResponse({"error": "unknown recording"}, status_code=400)
                return JSONResponse(
                    await _run_and_write(src, src.stem, lang, diarize, speakers, job))

            if file is None or not file.filename:
                return JSONResponse({"error": "no file provided"}, status_code=400)
            safe = Path(file.filename).name
            with tempfile.TemporaryDirectory() as td:
                src = Path(td) / safe
                with open(src, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                body = await _run_and_write(src, Path(safe).stem, lang, diarize,
                                            speakers, job)
            return JSONResponse(body)
        except Exception as exc:
            log.exception("Transcription failed")
            _set_stage(job, "failed")
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"},
                                status_code=500)

    def _open_capture():
        """Build the on-demand capture device. Returns (instance, ":N")."""
        if capture_factory is None:
            raise RuntimeError("automatic system-audio capture needs macOS 14.2 or later")
        instance = capture_factory()
        return instance, instance.start()

    @app.get("/api/recordings/unfinished")
    def unfinished():
        """Recordings on disk that were never transcribed.

        A crash, a restart, or the terminal being closed mid-meeting ends a
        recording without anyone stopping it. The audio survives — both
        recorders write as they go — but nothing points at it, which is how a
        93MB recording sat unnoticed for two days. Surfaced in the Record panel
        rather than a screen of its own.
        """
        found = recorder.unfinished_recordings(cfg.output_dir / "recordings",
                                               cfg.output_dir)
        for entry in found:
            try:
                # An interrupted recording claims zero samples in its header.
                if recorder.repair_wav_header(Path(entry["path"])):
                    log.info("Repaired the header of an interrupted recording: %s",
                             Path(entry["path"]).name)
            except Exception:
                log.exception("Could not repair %s", entry["path"])
        return {"recordings": found}

    @app.get("/api/devices")
    def devices():
        found, error = [], None
        try:
            found = recorder.list_input_devices(sys.platform)
        except Exception as exc:
            log.exception("Listing input devices failed")
            error = str(exc)
        # Hide the transient capture device: it is offered as "automatic".
        found = [d for d in found if d["name"].strip() != CAPTURE_DEVICE_NAME]
        if capture_factory is not None:
            found.insert(0, {"id": AUTO_DEVICE_ID, "name": AUTO_DEVICE_NAME})
        body: dict = {"devices": found}
        if error:
            body["error"] = error
        return body

    @app.post("/api/record/check")
    def record_check(device: str = Form(...), name: str = Form(default="")):
        """Sample the device before committing to a meeting-length recording.

        A virtual device like BlackHole 2ch opens happily and returns silence
        when nothing is routed into it, which is indistinguishable from success
        until the transcript comes back empty.
        """
        instance = None
        try:
            if device == AUTO_DEVICE_ID:
                instance, device = _open_capture()
                name = name or AUTO_DEVICE_NAME
            peak = probe(device, sys.platform)
        except Exception as exc:
            log.exception("Could not probe input device")
            return JSONResponse({"error": str(exc)}, status_code=200)
        finally:
            # Only a probe: hand the device back rather than holding the tap.
            if instance is not None:
                instance.stop()
        silent = recorder.is_silent(peak)
        body = {"peak_dbfs": round(peak, 1), "silent": silent}
        if silent:
            body["hint"] = recorder.silence_hint(name)
        return body

    @app.post("/api/record/start")
    def record_start(device: str = Form(...), name: str = Form(default="")):
        rec_id = uuid.uuid4().hex[:8]
        dest = cfg.output_dir / "recordings"
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / f"recording-{time.strftime('%Y%m%d-%H%M%S')}.wav"
        instance = None
        try:
            if device == AUTO_DEVICE_ID:
                instance, device = _open_capture()
                name = name or AUTO_DEVICE_NAME
            proc = recorder.start_recording(device, out, sys.platform)
        except Exception as exc:
            log.exception("Could not start recording")
            if instance is not None:
                instance.stop()      # never leave a tap behind on failure
            return JSONResponse({"error": str(exc)}, status_code=500)
        engine_used = recorder.recorder_name(proc)
        if engine_used != "native":
            log.warning("Recording %s is being made by ffmpeg. Its capture "
                        "tolerates about 10ms of delay and loses audio on a busy "
                        "machine; build the native recorder with "
                        "scripts/build-native.sh.", out.name)
        recordings[rec_id] = {"proc": proc, "path": out, "name": name,
                              "capture": instance, "recorder": engine_used}
        return {"rec_id": rec_id, "path": str(out), "recorder": engine_used}

    @app.get("/api/record/level")
    def record_level(rec_id: str):
        """Peak level and completeness so far, for the live meter.

        Measured from the growing file rather than by opening the device a
        second time — some inputs allow only one reader. Loss is reported here
        as well as at stop: a meeting that is losing audio has to be visible
        while it can still be rescued, not discovered afterwards.
        """
        entry = recordings.get(rec_id)
        if entry is None:
            return JSONResponse({"error": "unknown recording"}, status_code=404)
        body: dict = {"peak_dbfs": None, "silent": False}
        try:
            peak = recorder.peak_dbfs(entry["path"], tail_seconds=2.0)
            body = {"peak_dbfs": round(peak, 1), "silent": recorder.is_silent(peak)}
        except Exception:
            # The file may not have its header yet in the first moments.
            pass
        progress = recorder.read_progress(entry["path"])
        if "dropped" in progress:
            body["dropped"] = round(float(progress["dropped"]), 3)
            body["captured_seconds"] = progress.get("captured_seconds")
            if progress.get("stalled"):
                # A device that has stopped is not the same problem as one
                # losing buffers, and the thing to do about it is different.
                body["stalled"] = True
                body["hint"] = recorder.stalled_hint(
                    float(progress.get("stall_seconds") or 0.0),
                    entry.get("name", ""))
            elif body["dropped"] > recorder.DROPPED_FRACTION_LIMIT:
                body["hint"] = recorder.dropped_hint(body["dropped"],
                                                     entry.get("recorder", "ffmpeg"))
        return body

    @app.post("/api/record/stop")
    def record_stop(rec_id: str = Form(...)):
        entry = recordings.pop(rec_id, None)
        if entry is None:
            return JSONResponse({"error": "unknown recording"}, status_code=404)
        stats: dict = {}
        try:
            stats = recorder.stop_recording(entry["proc"]) or {}
        except Exception as exc:
            log.exception("Could not stop the recorder")
            return JSONResponse({"error": str(exc)}, status_code=500)
        finally:
            recorder.progress_path(entry["path"]).unlink(missing_ok=True)
            if entry.get("capture") is not None:
                # The tap lives in this process; releasing it removes the device
                # from every app's input list again. It must not depend on the
                # recorder having shut down cleanly, or a failed stop strands a
                # device that only a restart will clear. Swallowed rather than
                # raised: this runs on the way out, including out of an error.
                try:
                    entry["capture"].stop()
                except Exception:
                    log.exception("Could not release the capture device")
        path = entry["path"]
        if not path.exists() or path.stat().st_size == 0:
            return JSONResponse({"error": "recording produced no audio"}, status_code=500)
        # Size is not evidence of content: 24 minutes of silence is 46MB.
        peak = recorder.peak_dbfs(path)
        silent = recorder.is_silent(peak)
        # Nor is loudness evidence of completeness. A throttled capture drops
        # most of its buffers and still peaks like a healthy recording. The
        # native recorder counts what CoreAudio never handed it; ffmpeg cannot,
        # so its recordings fall back to measuring filled-in silence.
        used = entry.get("recorder", "ffmpeg")
        dropped = recorder.reported_dropped(stats, path)
        body = {"path": str(path), "bytes": path.stat().st_size,
                "peak_dbfs": round(peak, 1), "silent": silent,
                "dropped": round(dropped, 3), "recorder": used}
        if stats.get("captured_seconds") is not None:
            body["captured_seconds"] = stats["captured_seconds"]
        if silent:
            body["hint"] = recorder.silence_hint(entry.get("name", ""))
        elif dropped > recorder.DROPPED_FRACTION_LIMIT:
            log.warning("Recording %s lost %.0f%% of its audio to the capture "
                        "layer (%s)", path.name, dropped * 100, used)
            body["hint"] = recorder.dropped_hint(dropped, used)
        return body

    @app.get("/api/file")
    def get_file(path: str):
        p = Path(path).resolve()
        allowed = cfg.output_dir.resolve()
        if allowed not in p.parents:
            return JSONResponse({"error": "forbidden"}, status_code=403)
        if not p.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(p)

    # serve static assets (app.css)
    from fastapi.staticfiles import StaticFiles
    (HERE / "static").mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    return app

setup_logging(LOG_DIR)
app = create_app()
