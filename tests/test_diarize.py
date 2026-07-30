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

def test_num_speakers_is_passed_to_pipeline():
    seen = {}
    def factory(hf_token):
        def pipeline(wav, **kw):
            seen.update(kw)
            return FakeDiarizeOutput()
        return pipeline
    diarize_wav(Path("x.wav"), "tok", pipeline_factory=factory, num_speakers=2)
    assert seen == {"num_speakers": 2}

def test_num_speakers_omitted_when_not_given():
    seen = {"called": False}
    def factory(hf_token):
        def pipeline(wav, **kw):
            seen["called"] = True
            seen["kw"] = kw
            return FakeDiarizeOutput()
        return pipeline
    diarize_wav(Path("x.wav"), "tok", pipeline_factory=factory)
    assert seen["called"] and seen["kw"] == {}


# ---- which device the pipeline runs on ---------------------------------------
# pyannote on the CPU was measured at 119.5s for two minutes of audio against
# 15.1s on Metal — nearly half of the whole pipeline's runtime. The GPU is
# therefore tried first, but never at the cost of losing speaker labels.

import engine.diarize as D

class FakePipelineWithDevice:
    """Records .to() the way pyannote's Pipeline does (returns self)."""
    def __init__(self, fail_on=()):
        self.devices = []
        self.fail_on = fail_on
    def to(self, device):
        self.devices.append(str(device))
        return self
    def __call__(self, wav, **kw):
        if self.devices and self.devices[-1] in self.fail_on:
            raise RuntimeError(f"op not implemented for {self.devices[-1]}")
        return FakeDiarizeOutput()

def test_gpu_is_preferred_when_available(monkeypatch):
    monkeypatch.setattr(D, "_devices", lambda: ["mps", "cpu"])
    made = []
    def factory(hf_token):
        p = FakePipelineWithDevice(); made.append(p); return p
    turns = diarize_wav(Path("x.wav"), "tok", pipeline_factory=factory)
    assert made[0].devices == ["mps"]        # went straight to the GPU
    assert len(made) == 1                    # and needed no second attempt
    assert [t.speaker for t in turns] == ["SPEAKER_00", "SPEAKER_01"]

def test_gpu_failure_falls_back_to_cpu_and_keeps_labels(monkeypatch):
    monkeypatch.setattr(D, "_devices", lambda: ["mps", "cpu"])
    made = []
    def factory(hf_token):
        p = FakePipelineWithDevice(fail_on=("mps",)); made.append(p); return p
    turns = diarize_wav(Path("x.wav"), "tok", pipeline_factory=factory)
    assert [p.devices[-1] for p in made] == ["mps", "cpu"]
    assert [t.speaker for t in turns] == ["SPEAKER_00", "SPEAKER_01"], \
        "an MPS failure must cost speed, not speaker labels"

def test_failure_on_every_device_still_degrades_to_empty(monkeypatch):
    monkeypatch.setattr(D, "_devices", lambda: ["mps", "cpu"])
    def factory(hf_token):
        return FakePipelineWithDevice(fail_on=("mps", "cpu"))
    assert diarize_wav(Path("x.wav"), "tok", pipeline_factory=factory) == []

def test_pipelines_without_a_to_method_still_run(monkeypatch):
    # The fakes above and any plain callable have no .to(); moving to a device
    # must be optional, not required.
    monkeypatch.setattr(D, "_devices", lambda: ["mps", "cpu"])
    turns = diarize_wav(Path("x.wav"), "tok", pipeline_factory=fake_factory)
    assert [t.speaker for t in turns] == ["SPEAKER_00", "SPEAKER_01"]

def test_devices_lists_a_cpu_fallback_after_any_accelerator():
    devices = D._devices()
    assert devices[-1] == "cpu", devices
    assert len(devices) == len(set(devices))
