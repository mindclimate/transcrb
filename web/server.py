import logging
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.concurrency import run_in_threadpool
from engine.config import load_config
from engine.pipeline import transcribe_file
from engine.output import write_outputs
from engine import record as recorder

log = logging.getLogger(__name__)
HERE = Path(__file__).parent

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

def create_app(cfg=None, runner=None) -> FastAPI:
    cfg = cfg or load_config(Path("config.toml"))
    runner = runner or transcribe_file
    app = FastAPI()
    recordings: dict[str, dict] = {}

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
        result = await run_in_threadpool(
            runner, src, cfg, lang_arg, diarize.lower() != "false",
            lambda stage: _set_stage(job, stage),
            n_spk,
        )
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

    @app.get("/api/devices")
    def devices():
        try:
            return {"devices": recorder.list_input_devices(sys.platform)}
        except Exception as exc:
            log.exception("Listing input devices failed")
            return JSONResponse({"devices": [], "error": str(exc)}, status_code=200)

    @app.post("/api/record/start")
    def record_start(device: str = Form(...)):
        rec_id = uuid.uuid4().hex[:8]
        dest = cfg.output_dir / "recordings"
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / f"recording-{time.strftime('%Y%m%d-%H%M%S')}.wav"
        try:
            proc = recorder.start_recording(device, out, sys.platform)
        except Exception as exc:
            log.exception("Could not start recording")
            return JSONResponse({"error": str(exc)}, status_code=500)
        recordings[rec_id] = {"proc": proc, "path": out}
        return {"rec_id": rec_id, "path": str(out)}

    @app.post("/api/record/stop")
    def record_stop(rec_id: str = Form(...)):
        entry = recordings.pop(rec_id, None)
        if entry is None:
            return JSONResponse({"error": "unknown recording"}, status_code=404)
        recorder.stop_recording(entry["proc"])
        path = entry["path"]
        if not path.exists() or path.stat().st_size == 0:
            return JSONResponse({"error": "recording produced no audio"}, status_code=500)
        return {"path": str(path), "bytes": path.stat().st_size}

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

app = create_app()
