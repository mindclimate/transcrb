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

class FakeDiarizeOutput:
    """pyannote 4.x shape: serialize() instead of itertracks()."""
    def serialize(self):
        return {"diarization": [
            {"start": 2.3, "end": 4.0, "speaker": "SPEAKER_01"},
            {"start": 0.0, "end": 2.2, "speaker": "SPEAKER_00"},
        ]}

def test_reads_pyannote_4x_serialize_output():
    def factory(hf_token):
        return lambda wav: FakeDiarizeOutput()
    turns = diarize_wav(Path("x.wav"), "tok", pipeline_factory=factory)
    assert [t.speaker for t in turns] == ["SPEAKER_00", "SPEAKER_01"]
    assert turns[0].start == 0.0 and turns[1].end == 4.0

def test_unknown_result_type_degrades_to_empty():
    def factory(hf_token):
        return lambda wav: object()   # neither API generation
    assert diarize_wav(Path("x.wav"), "tok", pipeline_factory=factory) == []
