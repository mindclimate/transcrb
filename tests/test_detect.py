from pathlib import Path
from engine.detect import detect_language

class FakeInfo:
    def __init__(self, lang, prob): self.language = lang; self.language_probability = prob

class FakeModel:
    def __init__(self, lang, prob): self._info = FakeInfo(lang, prob)
    def transcribe(self, wav, **kw): return (iter([]), self._info)

def test_detects_hebrew():
    lang, prob = detect_language(Path("x.wav"), FakeModel("he", 0.95))
    assert lang == "he" and prob == 0.95

def test_detects_english():
    lang, _ = detect_language(Path("x.wav"), FakeModel("en", 0.9))
    assert lang == "en"

def test_other_language_falls_back():
    lang, _ = detect_language(Path("x.wav"), FakeModel("ru", 0.99), fallback="en")
    assert lang == "en"

def test_low_confidence_falls_back():
    lang, prob = detect_language(Path("x.wav"), FakeModel("he", 0.3), fallback="en")
    assert lang == "en"
