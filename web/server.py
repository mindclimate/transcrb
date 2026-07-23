import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from engine.config import load_config
from engine.pipeline import transcribe_file
from engine.output import write_outputs

HERE = Path(__file__).parent

def create_app(cfg=None, runner=None) -> FastAPI:
    cfg = cfg or load_config(Path("config.toml"))
    runner = runner or transcribe_file
    app = FastAPI()

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/")
    def index():
        return FileResponse(HERE / "index.html")

    @app.post("/api/transcribe")
    async def transcribe(file: UploadFile = File(...),
                         lang: str = Form(default=""),
                         diarize: str = Form(default="true")):
        if not file.filename:
            return JSONResponse({"error": "no filename"}, status_code=400)
        safe = Path(file.filename).name
        name = Path(safe).stem
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / safe
            with open(src, "wb") as f:
                shutil.copyfileobj(file.file, f)
            lang_arg = lang or None
            result = await run_in_threadpool(
                runner, src, cfg, lang_arg, diarize.lower() != "false"
            )
        paths = write_outputs(result, cfg.output_dir, name, inbox=cfg.inbox)
        return JSONResponse({"name": name, "md": str(paths["md"]),
                             "json": str(paths["json"])})

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
