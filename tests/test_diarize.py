from pathlib import Path
from engine.types import SpeakerTurn
from engine.diarize import diarize_wav

class FakeTurn:
    def __init__(self, s, e): self.start = s; self.end = e

class FakeAnnotation:
    def itertracks(self, yield_label=False):
        yield FakeTurn(2.3, 4.0), None, "SPEAKER_01"
        yield FakeTurn(0.0, 2.2), None, "SPEAKER_00"

def fake_factory(hf_token):
    def pipeline(wav): return FakeAnnotation()
    return pipeline

def test_returns_sorted_turns():
    turns = diarize_wav(Path("x.wav"), "tok", pipeline_factory=fake_factory)
    assert [t.speaker for t in turns] == ["SPEAKER_00", "SPEAKER_01"]
    assert isinstance(turns[0], SpeakerTurn)
    assert turns[0].start == 0.0

def test_no_token_returns_empty():
    assert diarize_wav(Path("x.wav"), None) == []

def test_factory_error_degrades_to_empty():
    def boom(hf_token): raise RuntimeError("model download failed")
    assert diarize_wav(Path("x.wav"), "tok", pipeline_factory=boom) == []
