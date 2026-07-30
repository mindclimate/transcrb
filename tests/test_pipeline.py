from pathlib import Path
import numpy as np
import soundfile as sf
import engine.pipeline as P
from engine.detect import LangSpan
from engine.types import Word, SpeakerTurn, Segment
from engine.config import Config

def _cfg():
    return Config(output_dir=Path("out"), inbox=None, hf_token="tok",
                  compute_type="int8", fallback_language="en")

def _fake_normalize(src, dst):
    """Stand in for ffmpeg by writing real silence.

    The pipeline reads the normalized file once and slices every span out of
    those samples, so this has to be an actual readable wav rather than a path.
    """
    sf.write(str(dst), np.zeros(16000, dtype="float32"), 16000, subtype="PCM_16")
    return dst

def test_pipeline_wires_stages(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "normalize_audio", _fake_normalize)
    monkeypatch.setattr(P, "load_detection_model", lambda ct: "detector")
    monkeypatch.setattr(P, "detect_language_spans",
                        lambda audio, rate, model, fallback: [LangSpan(0.0, 2.0, "he")])
    monkeypatch.setattr(P, "load_model", lambda lang, md, ct: object())
    monkeypatch.setattr(P, "transcribe_wav",
                        lambda audio, model, lang: ([Word(0.0, 2.0, "שלום")], 2.0))
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
    def _detect(*a, **k):
        called["detect"] = True
        return [LangSpan(0.0, 1.0, "en")]
    monkeypatch.setattr(P, "normalize_audio", _fake_normalize)
    monkeypatch.setattr(P, "load_detection_model", lambda ct: "detector")
    monkeypatch.setattr(P, "detect_language_spans", _detect)
    monkeypatch.setattr(P, "load_model", lambda lang, md, ct: object())
    monkeypatch.setattr(P, "transcribe_wav",
                        lambda audio, model, lang: ([Word(0.0, 1.0, "hi")], 1.0))
    src = tmp_path / "m.wav"; src.write_bytes(b"x")
    result = P.transcribe_file(src, _cfg(), lang="en", diarize=False)
    assert called["detect"] is False          # explicit lang skips detection
    assert result.segments[0].speaker == "SPEAKER_00"   # no diarization -> single speaker

def test_english_autodetect_loads_large_v3_once(monkeypatch, tmp_path):
    # The detection model is separate and small; the transcription model must
    # still only be loaded a single time.
    loads = []
    monkeypatch.setattr(P, "normalize_audio", _fake_normalize)
    monkeypatch.setattr(P, "load_detection_model", lambda ct: "detector")
    monkeypatch.setattr(P, "detect_language_spans",
                        lambda audio, rate, model, fallback: [LangSpan(0.0, 1.0, "en")])
    monkeypatch.setattr(P, "load_model", lambda lang, md, ct: loads.append(lang) or object())
    monkeypatch.setattr(P, "transcribe_wav", lambda audio, model, lang: ([Word(0.0, 1.0, "hi")], 1.0))
    monkeypatch.setattr(P, "diarize_wav", lambda wav, tok, num_speakers=None: [])
    src = tmp_path / "m.wav"; src.write_bytes(b"x")
    P.transcribe_file(src, _cfg(), lang=None, diarize=False)
    assert loads == ["en"]        # exactly one load, not two

def test_hebrew_autodetect_loads_hebrew_model(monkeypatch, tmp_path):
    loads = []
    monkeypatch.setattr(P, "normalize_audio", _fake_normalize)
    monkeypatch.setattr(P, "load_detection_model", lambda ct: "detector")
    monkeypatch.setattr(P, "detect_language_spans",
                        lambda audio, rate, model, fallback: [LangSpan(0.0, 1.0, "he")])
    monkeypatch.setattr(P, "load_model", lambda lang, md, ct: loads.append(lang) or object())
    monkeypatch.setattr(P, "transcribe_wav", lambda audio, model, lang: ([Word(0.0, 1.0, "שלום")], 1.0))
    monkeypatch.setattr(P, "diarize_wav", lambda wav, tok, num_speakers=None: [])
    src = tmp_path / "m.wav"; src.write_bytes(b"x")
    P.transcribe_file(src, _cfg(), lang=None, diarize=False)
    # Language ID has its own small model, so a Hebrew meeting never loads
    # large-v3 at all — only the ivrit model that will actually be used.
    assert loads == ["he"]


# ---- mixed-language recordings ----------------------------------------------
# Auto-detect now scans for language spans instead of labelling the whole file
# from its opening 30 seconds. Each span goes to the model that matches it, so
# Hebrew is transcribed by the Hebrew model rather than translated by the
# English one.

from engine.types import Word as W

def _stub_stages(monkeypatch, spans, *, duration=90.0):
    """Wire the pipeline to fake audio, detection and diarization."""
    loaded = []
    spanned = []
    monkeypatch.setattr(P, "normalize_audio", _fake_normalize)
    monkeypatch.setattr(P, "load_detection_model", lambda ct: "detector")
    monkeypatch.setattr(P, "probe_duration", lambda wav: duration)
    monkeypatch.setattr(P, "detect_language_spans",
                        lambda audio, rate, model, fallback: spans)
    monkeypatch.setattr(P, "diarize_wav", lambda wav, tok, num_speakers=None: [])

    def _load(language, models_dir, compute_type):
        loaded.append(language)
        return f"model:{language}"

    def _span(audio, rate, model, language, start, end):
        spanned.append({"model": model, "lang": language, "start": start, "end": end})
        return [W(start, end, f"{language}-text", lang=language)]

    monkeypatch.setattr(P, "load_model", _load)
    monkeypatch.setattr(P, "transcribe_span", _span)
    monkeypatch.setattr(P, "transcribe_wav",
                        lambda audio, model, lang: ([W(0.0, duration, "whole", lang=lang)],
                                                  duration))
    return loaded, spanned

def test_mixed_spans_each_go_to_their_own_model(monkeypatch, tmp_path):
    spans = [LangSpan(0.0, 30.0, "en"), LangSpan(30.0, 60.0, "he"),
             LangSpan(60.0, 90.0, "en")]
    loaded, spanned = _stub_stages(monkeypatch, spans)
    src = tmp_path / "m.wav"; src.write_bytes(b"x")
    result = P.transcribe_file(src, _cfg(), lang=None, diarize=False)
    assert len(spanned) == 3
    assert {s["lang"] for s in spanned} == {"en", "he"}
    assert {s["model"] for s in spanned} == {"model:en", "model:he"}
    assert result.language == "en+he"

def test_words_come_back_in_timeline_order_across_languages(monkeypatch, tmp_path):
    spans = [LangSpan(0.0, 30.0, "en"), LangSpan(30.0, 60.0, "he"),
             LangSpan(60.0, 90.0, "en")]
    _stub_stages(monkeypatch, spans)
    src = tmp_path / "m.wav"; src.write_bytes(b"x")
    result = P.transcribe_file(src, _cfg(), lang=None, diarize=False)
    starts = [w.start for w in result.words]
    assert starts == sorted(starts), "words must be ordered by time, not by language"
    assert [w.lang for w in result.words] == ["en", "he", "en"]

def test_the_english_spans_reuse_the_detection_model(monkeypatch, tmp_path):
    # large-v3 does the detecting and the English transcribing; loading it twice
    # would cost a second multi-GB load for the same weights.
    spans = [LangSpan(0.0, 30.0, "en"), LangSpan(30.0, 60.0, "he")]
    loaded, _spanned = _stub_stages(monkeypatch, spans)
    src = tmp_path / "m.wav"; src.write_bytes(b"x")
    P.transcribe_file(src, _cfg(), lang=None, diarize=False)
    assert loaded.count("en") == 1

def test_a_single_language_scan_uses_the_whole_file_path(monkeypatch, tmp_path):
    # One span covering everything is the common case and must stay as cheap as
    # it was before spans existed: one transcription of the whole file.
    spans = [LangSpan(0.0, 90.0, "he")]
    _loaded, spanned = _stub_stages(monkeypatch, spans)
    src = tmp_path / "m.wav"; src.write_bytes(b"x")
    result = P.transcribe_file(src, _cfg(), lang=None, diarize=False)
    assert spanned == []                      # no per-span slicing needed
    assert result.language == "he"
    assert result.words[0].text == "whole"

def test_mixed_model_names_are_both_reported(monkeypatch, tmp_path):
    spans = [LangSpan(0.0, 30.0, "en"), LangSpan(30.0, 60.0, "he")]
    _stub_stages(monkeypatch, spans)
    src = tmp_path / "m.wav"; src.write_bytes(b"x")
    result = P.transcribe_file(src, _cfg(), lang=None, diarize=False)
    assert "large-v3" in result.model and "ivrit" in result.model

def test_progress_names_the_language_being_transcribed(monkeypatch, tmp_path):
    spans = [LangSpan(0.0, 30.0, "en"), LangSpan(30.0, 60.0, "he")]
    _stub_stages(monkeypatch, spans)
    src = tmp_path / "m.wav"; src.write_bytes(b"x")
    stages = []
    P.transcribe_file(src, _cfg(), lang=None, diarize=False,
                      progress=stages.append)
    assert any("he" in s for s in stages), stages
    assert "detecting language" in stages

def test_explicit_language_never_scans_for_spans(monkeypatch, tmp_path):
    spans = [LangSpan(0.0, 90.0, "en")]
    _stub_stages(monkeypatch, spans)
    scanned = []
    monkeypatch.setattr(P, "detect_language_spans",
                        lambda *a, **k: scanned.append(1) or spans)
    src = tmp_path / "m.wav"; src.write_bytes(b"x")
    result = P.transcribe_file(src, _cfg(), lang="he", diarize=False)
    assert scanned == []
    assert result.language == "he"
