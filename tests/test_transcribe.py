from pathlib import Path
from engine.types import Word
from engine.transcribe import model_ref, transcribe_wav

def test_model_ref_hebrew_uses_local_ivrit_turbo(tmp_path):
    ref = model_ref("he", tmp_path)
    assert ref == str(tmp_path / "ivrit-whisper-turbo-ct2")

def test_model_ref_english_uses_large_v3_turbo(tmp_path):
    assert model_ref("en", tmp_path) == "large-v3-turbo"

def test_hebrew_falls_back_to_pre_turbo_weights_when_only_those_exist(tmp_path):
    # An install from before the turbo swap has the old directory and not the new
    # one. It must keep transcribing on what it has instead of pointing at a
    # directory that is not there.
    (tmp_path / "ivrit-whisper-ct2").mkdir()
    assert model_ref("he", tmp_path) == str(tmp_path / "ivrit-whisper-ct2")

def test_hebrew_prefers_turbo_when_both_are_present(tmp_path):
    (tmp_path / "ivrit-whisper-ct2").mkdir()
    (tmp_path / "ivrit-whisper-turbo-ct2").mkdir()
    assert model_ref("he", tmp_path) == str(tmp_path / "ivrit-whisper-turbo-ct2")

class FakeWord:
    def __init__(self, s, e, w): self.start = s; self.end = e; self.word = w

class FakeSeg:
    def __init__(self, words): self.words = words

class FakeInfo:
    duration = 2.0

class FakeModel:
    def transcribe(self, wav, **kw):
        segs = [FakeSeg([FakeWord(0.0, 1.0, "hello"), FakeWord(1.0, 2.0, "world")])]
        return iter(segs), FakeInfo()

def test_transcribe_wav_flattens_words():
    words, dur = transcribe_wav(Path("x.wav"), FakeModel(), "en")
    assert [w.text for w in words] == ["hello", "world"]
    assert isinstance(words[0], Word)
    assert dur == 2.0


# ---- transcribing one language span ------------------------------------------
# Each span is sent to the model that matches its language, so the words come
# back with timestamps relative to the span and have to be shifted into the
# timeline of the whole recording.

import numpy as np
import soundfile as sf
from engine.transcribe import transcribe_span

class _FakeSeg:
    def __init__(self, words):
        self.words = words

class _FakeWord:
    def __init__(self, start, end, word):
        self.start, self.end, self.word = start, end, word

class _FakeModel:
    """Returns words positioned relative to whatever audio it was handed."""
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio": audio, **kwargs})
        seconds = len(audio) / 16000
        return ([_FakeSeg([_FakeWord(0.0, seconds / 2, "first"),
                           _FakeWord(seconds / 2, seconds, "second")])],
                object())

def _samples(seconds, rate=16000):
    """The recording as the pipeline hands it over: samples read once, plus rate."""
    return np.zeros(int(seconds * rate), dtype="float32"), rate

def test_span_words_are_shifted_into_the_full_recording_timeline():
    audio, rate = _samples(60.0)
    words = transcribe_span(audio, rate, _FakeModel(), "he", 20.0, 30.0)
    assert words[0].start == 20.0            # not 0.0 — the span starts at 20s
    assert words[-1].end == 30.0

def test_span_words_are_tagged_with_the_span_language():
    audio, rate = _samples(30.0)
    words = transcribe_span(audio, rate, _FakeModel(), "he", 0.0, 10.0)
    assert {w.lang for w in words} == {"he"}

def test_only_the_span_audio_is_handed_to_the_model():
    audio, rate = _samples(60.0)
    model = _FakeModel()
    transcribe_span(audio, rate, model, "en", 10.0, 25.0)
    handed = model.calls[0]["audio"]
    assert abs(len(handed) / 16000 - 15.0) < 0.01     # 15s slice, not the file
    assert model.calls[0]["language"] == "en"

def test_a_span_covering_the_whole_file_keeps_original_timings():
    audio, rate = _samples(12.0)
    words = transcribe_span(audio, rate, _FakeModel(), "en", 0.0, 12.0)
    assert words[0].start == 0.0 and words[-1].end == 12.0

def test_the_samples_are_never_re_read_per_span():
    # The whole point of taking samples: two spans, no file access at all.
    audio, rate = _samples(60.0)
    model = _FakeModel()
    transcribe_span(audio, rate, model, "en", 0.0, 10.0)
    transcribe_span(audio, rate, model, "he", 10.0, 20.0)
    assert len(model.calls) == 2
    assert all(len(c["audio"]) == 10 * rate for c in model.calls)
