import soundfile as sf
from pathlib import Path
from engine.audio import normalize_audio, probe_duration

FIX = Path(__file__).parent / "fixtures"

def test_normalize_to_16k_mono(tmp_path):
    dst = tmp_path / "out.wav"
    normalize_audio(FIX / "tone_stereo_44k.wav", dst)
    info = sf.info(str(dst))
    assert info.samplerate == 16000
    assert info.channels == 1

def test_probe_duration(tmp_path):
    dst = tmp_path / "out.wav"
    normalize_audio(FIX / "tone_stereo_44k.wav", dst)
    assert 1.8 < probe_duration(dst) < 2.2

def test_missing_file_raises(tmp_path):
    import pytest
    with pytest.raises(RuntimeError):
        normalize_audio(tmp_path / "nope.m4a", tmp_path / "o.wav")
