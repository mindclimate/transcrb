import argparse
import logging
import sys
from pathlib import Path
from engine.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from engine.logs import setup_logging
from engine.pipeline import transcribe_file
from engine.output import write_outputs

AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".mp4", ".aac", ".flac", ".ogg", ".webm"}

log = logging.getLogger(__name__)

def _iter_audio(path: Path):
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    return [path]

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="transcribe")
    ap.add_argument("path")
    ap.add_argument("--lang", choices=["he", "en"], default=None)
    ap.add_argument("--no-diarize", action="store_true")
    ap.add_argument("--speakers", type=int, default=None,
                    help="how many speakers to expect (improves labelling)")
    ap.add_argument("--inbox", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = ap.parse_args(argv)
    setup_logging(PROJECT_ROOT / "logs")

    path = Path(args.path)
    if not path.exists():
        print(f"error: path not found: {path}", file=sys.stderr)
        return 1

    cfg = load_config(Path(args.config))
    out_dir = Path(args.out) if args.out else cfg.output_dir
    inbox = Path(args.inbox) if args.inbox else cfg.inbox

    files = _iter_audio(path)
    if not files:
        print(f"error: no audio files in {path}", file=sys.stderr)
        return 1

    for f in files:
        print(f"transcribing {f.name} ...")
        try:
            result = transcribe_file(f, cfg, lang=args.lang,
                                     diarize=not args.no_diarize,
                                     num_speakers=args.speakers,
                                     progress=lambda s: print(f"  {s}"))
        except Exception as e:  # loud, specific, per spec
            # The message goes to the person; the traceback goes to the log, so
            # a failure is still diagnosable after the terminal is closed.
            log.exception("Transcription failed for %s", f)
            print(f"error transcribing {f.name}: {e}", file=sys.stderr)
            return 1
        paths = write_outputs(result, out_dir, f.stem, inbox=inbox)
        print(f"  -> {paths['md']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
