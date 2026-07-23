"""Generate tiny synthetic audio fixtures. Run: python tests/fixtures/make_fixtures.py"""
import numpy as np, soundfile as sf
from pathlib import Path

here = Path(__file__).parent

sr44 = 44100
t44 = np.linspace(0, 2.0, int(sr44 * 2.0), endpoint=False)
tone44 = 0.2 * np.sin(2 * np.pi * 220 * t44).astype("float32")
sf.write(here / "tone_stereo_44k.wav", np.column_stack([tone44, tone44]), sr44)

sr = 16000
t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
tone = 0.2 * np.sin(2 * np.pi * 220 * t).astype("float32")
sf.write(here / "tone_mono_16k.wav", tone, sr)
print("fixtures written")
