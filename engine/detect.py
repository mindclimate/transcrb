import math
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {"he", "en"}

# Whisper decides a language from one 30s mel window, so a file gets one label
# and a meeting that switches language mid-way is transcribed entirely in the
# wrong one — worse, Whisper *translates* the other language instead of failing,
# producing a fluent transcript of words nobody said. The scan below splits the
# recording into single-language spans first.
#
# Two measurements shaped this (see docs/SETUP.md):
#   * A window's reported language reflects its opening moment, not its
#     majority, and a window straddling a switch still reports high confidence.
#     Confidence therefore cannot be used to find switches — only a short enough
#     hop can, so the scan probes uniformly instead of trying to be clever.
#   * The small `base` model is ~20x cheaper per probe than large-v3 *and* more
#     accurate here, scoring 19/19 on a clip that large-v3 read as pure English.
PROBE_WINDOW = 6.0       # audio given to each probe: enough context to be sure
PROBE_HOP = 2.0          # how far apart probes sit — this is the resolution
CONFIDENT = 0.5          # `base` is less emphatic than large-v3; junk is
                         # filtered by the supported-language check anyway
MIN_SPAN = 5.0           # shorter than this is detector noise, not a switch

@dataclass(frozen=True)
class LangSpan:
    start: float
    end: float
    lang: str

def detect_language(wav: Path, model, fallback: str = "en",
                    min_confidence: float = 0.6) -> tuple[str, float]:
    """The single dominant language of a file, from its opening window."""
    # faster-whisper returns (segments_generator, info) with language on `info`
    _segments, info = model.transcribe(str(wav), language=None)
    lang = info.language
    prob = float(info.language_probability)
    if lang not in SUPPORTED or prob < min_confidence:
        return fallback, prob
    return lang, prob

def _label(detect, start: float, end: float, confident: float,
           supported) -> str | None:
    """The language of one window, or None when the answer is not usable."""
    lang, prob = detect(start, end)
    if lang not in supported or prob < confident:
        return None
    return lang

def _coalesce(labelled: list[tuple[float, float, str]]) -> list[LangSpan]:
    spans: list[LangSpan] = []
    for start, end, lang in labelled:
        if spans and spans[-1].lang == lang and spans[-1].end == start:
            spans[-1] = LangSpan(spans[-1].start, end, lang)
        else:
            spans.append(LangSpan(start, end, lang))
    return spans

def _absorb_short_spans(spans: list[LangSpan], min_span: float) -> list[LangSpan]:
    """Give a too-short span to whichever neighbour is longer."""
    while len(spans) > 1:
        i = min(range(len(spans)), key=lambda j: spans[j].end - spans[j].start)
        if spans[i].end - spans[i].start >= min_span:
            break
        before = spans[i - 1] if i > 0 else None
        after = spans[i + 1] if i < len(spans) - 1 else None
        if before and after:
            target = before if (before.end - before.start) >= (after.end - after.start) \
                else after
        else:
            target = before or after
        assert target is not None
        spans[i] = LangSpan(spans[i].start, spans[i].end, target.lang)
        spans = _coalesce([(s.start, s.end, s.lang) for s in spans])
    return spans

def scan_language_spans(duration: float, detect, *,
                        window: float = PROBE_WINDOW,
                        hop: float = PROBE_HOP,
                        confident: float = CONFIDENT,
                        min_span: float = MIN_SPAN,
                        fallback: str = "en",
                        supported=SUPPORTED) -> list[LangSpan]:
    """Split `duration` seconds into single-language spans.

    `detect(start, end) -> (lang, prob)` is injected so this stays testable
    without audio. Probes are uniform and overlapping: each looks at `window`
    seconds but only labels the `hop` seconds at its start, because a window's
    verdict describes its opening rather than its content as a whole. `hop` is
    therefore both the cost and the boundary precision of the whole scan.
    """
    if duration <= 0:
        return [LangSpan(0.0, max(duration, 0.0), fallback)]

    labelled: list[tuple[float, float, str | None]] = []
    steps = max(1, math.ceil(duration / hop))
    for i in range(steps):
        start = i * hop
        end = min(start + hop, duration)
        if end <= start:
            break
        probe_end = min(start + window, duration)
        labelled.append((start, end,
                         _label(detect, start, probe_end, confident, supported)))

    known = [lang for _s, _e, lang in labelled if lang is not None]
    if not known:
        return [LangSpan(0.0, duration, fallback)]

    filled: list[tuple[float, float, str]] = []
    for start, end, lang in labelled:
        if lang is None:
            lang = filled[-1][2] if filled else known[0]
        filled.append((start, end, lang))

    return _absorb_short_spans(_coalesce(filled), min_span)

def detect_language_spans(audio, rate: int, model, fallback: str = "en",
                          **kwargs) -> list[LangSpan]:
    """Language spans of 16kHz mono samples, using Whisper's own language ID."""
    def detect(start: float, end: float) -> tuple[str, float]:
        chunk = audio[int(start * rate):int(end * rate)]
        lang, prob, _all = model.detect_language(audio=chunk)
        return lang, float(prob)

    return scan_language_spans(len(audio) / rate, detect, fallback=fallback,
                               **kwargs)
