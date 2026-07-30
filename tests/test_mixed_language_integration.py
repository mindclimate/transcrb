"""The test that actually proves the mixed-language bug is fixed.

Everything else about spans is unit-tested against a fake detector. This one
runs the real models over real speech that switches language, because the bug
was never in the span arithmetic — it was that Whisper, given one language for
the whole file, *translates* the other language instead of transcribing it. Only
real audio through real models can show that it no longer does.

Needs the multi-GB models and takes minutes, so it is opt-in:

    .venv/bin/python -m pytest -m slow tests/test_mixed_language_integration.py

Generate the fixture first with tests/fixtures/make_mixed_speech.sh (macOS).
"""
from pathlib import Path

import pytest

from engine.config import Config
from engine.pipeline import transcribe_file
from engine.transcribe import model_ref

FIXTURE = Path(__file__).parent / "fixtures" / "mixed_en_he.wav"
MODELS = Path(__file__).resolve().parent.parent / "models"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not FIXTURE.exists(),
                       reason="run tests/fixtures/make_mixed_speech.sh first"),
    # Whichever Hebrew weights the code resolves to, not a hardcoded directory:
    # this guard silently skipped the whole file when the model moved.
    pytest.mark.skipif(not Path(model_ref("he", MODELS)).exists(),
                       reason="Hebrew model not downloaded"),
]

def _has_hebrew(text: str) -> bool:
    return any("֐" <= c <= "׿" for c in text)

@pytest.fixture(scope="module")
def result():
    cfg = Config(output_dir=Path("out"), inbox=None, hf_token=None,
                 compute_type="int8", fallback_language="en")
    # Diarization needs a HuggingFace token and is irrelevant here.
    return transcribe_file(FIXTURE, cfg, lang=None, diarize=False)

def test_both_languages_are_reported(result):
    assert set(result.language.split("+")) == {"en", "he"}

def test_the_hebrew_is_transcribed_in_hebrew_not_translated(result):
    hebrew_segments = [s for s in result.segments if s.lang == "he"]
    assert hebrew_segments, "no Hebrew span was found at all"
    assert any(_has_hebrew(s.text) for s in hebrew_segments), \
        "Hebrew spans came back in Latin script — still being translated"

def test_the_english_stays_english(result):
    english = " ".join(s.text for s in result.segments if s.lang == "en")
    # Word timestamps split tokens unpredictably ("stand -up", "stand up"), so
    # compare on letters alone rather than on exact spelling.
    letters = "".join(c for c in english.lower() if c.isalnum())
    assert "standup" in letters
    assert "finishedtherecorderwork" in letters
    assert not _has_hebrew(english)

def test_no_segment_mixes_scripts(result):
    for seg in result.segments:
        latin = any(c.isascii() and c.isalpha() for c in seg.text)
        assert not (latin and _has_hebrew(seg.text)), \
            f"segment mixes scripts: {seg.text!r}"

def test_both_models_are_credited(result):
    assert "large-v3" in result.model
    assert "ivrit" in result.model
