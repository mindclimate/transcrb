"""Logging that survives the window being closed.

Transcrb runs from a terminal that the person using it closes when they are
done. Everything interesting — a failed transcription's traceback, a warning
that speaker labelling quietly fell back to the CPU and took four times longer
— went to that window's stdout and nowhere else, so by the time anyone asked
what went wrong there was nothing left to read. These handlers add a file
alongside the console output.
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FILENAME = "transcrb.log"
MAX_BYTES = 2_000_000
BACKUPS = 3

# Handlers this module installed, so repeated calls replace rather than stack.
_INSTALLED_MARKER = "_transcrb_installed"

# Third-party per-request chatter at INFO buries our own lines. The HuggingFace
# client logs every model-metadata GET; that is noise in a transcription log.
_QUIET = ("httpx", "httpcore", "urllib3", "filelock", "fsspec")

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

def setup_logging(log_dir: Path | str, max_bytes: int = MAX_BYTES,
                  backups: int = BACKUPS, level: int = logging.INFO) -> Path:
    """Send warnings and above to the console, everything to a capped file.

    Idempotent: calling it again replaces the handlers it installed before,
    rather than logging every line twice.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / LOG_FILENAME

    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _INSTALLED_MARKER, False)]:
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT)

    # delay=True so importing this module does not create an empty file for a
    # run that never logs anything.
    to_file = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups,
                                  encoding="utf-8", delay=True)
    to_file.setFormatter(formatter)
    to_file.setLevel(level)

    # StreamHandler() resolves sys.stderr on each emit, so the terminal keeps
    # showing what it always showed.
    to_console = logging.StreamHandler()
    to_console.setFormatter(formatter)
    to_console.setLevel(level)

    for handler in (to_file, to_console):
        setattr(handler, _INSTALLED_MARKER, True)
        root.addHandler(handler)

    root.setLevel(min(level, root.level or level))
    for name in _QUIET:
        logging.getLogger(name).setLevel(logging.WARNING)
    return path
