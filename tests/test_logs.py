import logging

from engine.logs import LOG_FILENAME, setup_logging

def test_writes_a_log_file_under_the_given_directory(tmp_path):
    setup_logging(tmp_path)
    logging.getLogger("web.server").warning("diarization fell back to the cpu")
    logging.shutdown()
    text = (tmp_path / LOG_FILENAME).read_text(encoding="utf-8")
    assert "diarization fell back to the cpu" in text

def test_records_the_whole_traceback_not_just_the_message(tmp_path):
    """The reason this exists: a failed transcription's traceback went to a
    terminal that was later closed, leaving nothing to diagnose."""
    setup_logging(tmp_path)
    try:
        raise RuntimeError("model exploded")
    except RuntimeError:
        logging.getLogger("web.server").exception("Transcription failed")
    logging.shutdown()
    text = (tmp_path / LOG_FILENAME).read_text(encoding="utf-8")
    assert "Traceback" in text
    assert "model exploded" in text

def test_calling_it_twice_does_not_double_every_line(tmp_path):
    setup_logging(tmp_path)
    setup_logging(tmp_path)
    logging.getLogger("web.server").warning("once")
    logging.shutdown()
    lines = [l for l in (tmp_path / LOG_FILENAME).read_text(encoding="utf-8").splitlines()
             if "once" in l]
    assert len(lines) == 1

def test_still_logs_to_the_console(tmp_path, capsys):
    """The terminal stays useful — the file is an addition, not a replacement."""
    setup_logging(tmp_path)
    logging.getLogger("web.server").warning("visible in the window too")
    assert "visible in the window too" in capsys.readouterr().err

def test_creates_the_directory_when_missing(tmp_path):
    target = tmp_path / "nested" / "logs"
    setup_logging(target)
    logging.getLogger("web.server").warning("hello")
    logging.shutdown()
    assert (target / LOG_FILENAME).is_file()

def test_caps_the_file_so_it_cannot_fill_the_disk(tmp_path):
    setup_logging(tmp_path, max_bytes=2_000, backups=1)
    for i in range(500):
        logging.getLogger("web.server").warning("a fairly long line to force rotation %d", i)
    logging.shutdown()
    written = sorted(p.name for p in tmp_path.glob(f"{LOG_FILENAME}*"))
    assert written == [LOG_FILENAME, f"{LOG_FILENAME}.1"]
    for p in tmp_path.glob(f"{LOG_FILENAME}*"):
        assert p.stat().st_size < 10_000
