from pathlib import Path

SUPPORTED = {"he", "en"}

def detect_language(wav: Path, model, fallback: str = "en",
                    min_confidence: float = 0.6) -> tuple[str, float]:
    # faster-whisper returns (segments_generator, info) with language on `info`
    _segments, info = model.transcribe(str(wav), language=None)
    lang = info.language
    prob = float(info.language_probability)
    if lang not in SUPPORTED or prob < min_confidence:
        return fallback, prob
    return lang, prob
