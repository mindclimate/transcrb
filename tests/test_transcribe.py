from pathlib import Path
from engine.types import Word
from engine.transcribe import model_ref, transcribe_wav

def test_model_ref_hebrew_uses_local_ivrit(tmp_path):
    ref = model_ref("he", tmp_path)
    assert ref == str(tmp_path / "ivrit-whisper-ct2")

def test_model_ref_english_uses_large_v3(tmp_path):
    assert model_ref("en", tmp_path) == "large-v3"

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
