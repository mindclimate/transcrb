from pathlib import Path
import engine.pipeline as P
from engine.types import Word, SpeakerTurn, Segment
from engine.config import Config

def _cfg():
    return Config(output_dir=Path("out"), inbox=None, hf_token="tok",
                  compute_type="int8", fallback_language="en")

def test_pipeline_wires_stages(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "normalize_audio", lambda src, dst: dst)
    monkeypatch.setattr(P, "detect_language", lambda wav, model, fallback: ("he", 0.9))
    monkeypatch.setattr(P, "load_model", lambda lang, md, ct: object())
    monkeypatch.setattr(P, "transcribe_wav",
                        lambda wav, model, lang: ([Word(0.0, 2.0, "שלום")], 2.0))
    monkeypatch.setattr(P, "diarize_wav",
                        lambda wav, tok, num_speakers=None: [SpeakerTurn(0.0, 2.0, "SPEAKER_00")])
    src = tmp_path / "meeting.m4a"
    src.write_bytes(b"x")
    result = P.transcribe_file(src, _cfg(), lang=None, diarize=True)
    assert result.language == "he"
    assert result.segments[0].text == "שלום"
    assert result.segments[0].speaker == "SPEAKER_00"
    assert "ivrit" in result.model

def test_pipeline_respects_explicit_lang_and_no_diarize(monkeypatch, tmp_path):
    called = {"detect": False}
    def _detect(*a, **k): called["detect"] = True; return ("en", 1.0)
    monkeypatch.setattr(P, "normalize_audio", lambda src, dst: dst)
    monkeypatch.setattr(P, "detect_language", _detect)
    monkeypatch.setattr(P, "load_model", lambda lang, md, ct: object())
    monkeypatch.setattr(P, "transcribe_wav",
                        lambda wav, model, lang: ([Word(0.0, 1.0, "hi")], 1.0))
    src = tmp_path / "m.wav"; src.write_bytes(b"x")
    result = P.transcribe_file(src, _cfg(), lang="en", diarize=False)
    assert called["detect"] is False          # explicit lang skips detection
    assert result.segments[0].speaker == "SPEAKER_00"   # no diarization -> single speaker

def test_english_autodetect_reuses_detection_model(monkeypatch, tmp_path):
    # large-v3 serves both detection and English transcription; loading it
    # twice would cost a second multi-GB load.
    loads = []
    monkeypatch.setattr(P, "normalize_audio", lambda src, dst: dst)
    monkeypatch.setattr(P, "detect_language", lambda wav, model, fallback: ("en", 0.99))
    monkeypatch.setattr(P, "load_model", lambda lang, md, ct: loads.append(lang) or object())
    monkeypatch.setattr(P, "transcribe_wav", lambda wav, model, lang: ([Word(0.0, 1.0, "hi")], 1.0))
    monkeypatch.setattr(P, "diarize_wav", lambda wav, tok, num_speakers=None: [])
    src = tmp_path / "m.wav"; src.write_bytes(b"x")
    P.transcribe_file(src, _cfg(), lang=None, diarize=False)
    assert loads == ["en"]        # exactly one load, not two

def test_hebrew_autodetect_loads_hebrew_model(monkeypatch, tmp_path):
    loads = []
    monkeypatch.setattr(P, "normalize_audio", lambda src, dst: dst)
    monkeypatch.setattr(P, "detect_language", lambda wav, model, fallback: ("he", 0.99))
    monkeypatch.setattr(P, "load_model", lambda lang, md, ct: loads.append(lang) or object())
    monkeypatch.setattr(P, "transcribe_wav", lambda wav, model, lang: ([Word(0.0, 1.0, "שלום")], 1.0))
    monkeypatch.setattr(P, "diarize_wav", lambda wav, tok, num_speakers=None: [])
    src = tmp_path / "m.wav"; src.write_bytes(b"x")
    P.transcribe_file(src, _cfg(), lang=None, diarize=False)
    assert loads == ["en", "he"]  # detection model, then the ivrit model
