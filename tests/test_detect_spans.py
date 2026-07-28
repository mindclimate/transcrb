"""Splitting a recording into language spans.

A standup that opens in English and switches to Hebrew used to be labelled `en`
for its whole length, and Whisper then *translated* the Hebrew into English
rather than transcribing it — a transcript that reads as fluent and complete
while silently misrepresenting what was said.

The fake detector below models what Whisper measurably does: **it reports the
language at the start of the window it is given**, at high confidence, even when
the rest of the window is another language. An earlier version of this scan
assumed a straddling window would betray itself with low confidence; it does
not, which is why the scan probes uniformly at a short hop instead.
"""
from engine.detect import LangSpan, scan_language_spans

def _timeline(*regions):
    """A fake detector over (start, end, lang, prob) regions.

    Reports the language of whichever region the window *starts* in — the
    measured behaviour of Whisper's language ID.
    """
    def detect(start, end):
        for r_start, r_end, lang, prob in regions:
            if r_start <= start < r_end:
                return lang, prob
        return "en", 0.0
    return detect

def test_a_single_language_file_is_one_span():
    spans = scan_language_spans(120.0, _timeline((0, 120, "en", 0.98)))
    assert spans == [LangSpan(0.0, 120.0, "en")]

def test_a_hebrew_only_file_is_one_hebrew_span():
    spans = scan_language_spans(90.0, _timeline((0, 90, "he", 0.99)))
    assert spans == [LangSpan(0.0, 90.0, "he")]

def test_a_switch_on_a_window_boundary_produces_two_spans():
    spans = scan_language_spans(60.0, _timeline((0, 30, "en", 0.98),
                                                (30, 60, "he", 0.98)))
    assert [(s.lang) for s in spans] == ["en", "he"]
    assert spans[0].start == 0.0 and spans[-1].end == 60.0

def test_a_switch_is_located_within_the_probe_hop():
    # A switch at 40s must be found to within the hop, since that is the only
    # thing bounding boundary error now.
    spans = scan_language_spans(90.0, _timeline((0, 40, "en", 0.98),
                                                (40, 90, "he", 0.98)))
    assert [s.lang for s in spans] == ["en", "he"]
    boundary = spans[0].end
    assert abs(boundary - 40.0) <= 2.0, f"boundary off by {abs(boundary-40):.1f}s"

def test_spans_tile_the_recording_without_gaps_or_overlap():
    spans = scan_language_spans(100.0, _timeline((0, 35, "en", 0.98),
                                                 (35, 70, "he", 0.98),
                                                 (70, 100, "en", 0.98)))
    assert spans[0].start == 0.0
    assert spans[-1].end == 100.0
    for earlier, later in zip(spans, spans[1:]):
        assert earlier.end == later.start

def test_three_alternating_spans_are_all_kept():
    spans = scan_language_spans(100.0, _timeline((0, 35, "en", 0.98),
                                                 (35, 70, "he", 0.98),
                                                 (70, 100, "en", 0.98)))
    assert [s.lang for s in spans] == ["en", "he", "en"]

def test_an_unconfident_window_does_not_invent_a_span():
    # Whisper reporting 0.4 on a noisy window must not carve the file up; the
    # surrounding language wins.
    spans = scan_language_spans(90.0, _timeline((0, 30, "en", 0.98),
                                                (30, 60, "he", 0.40),
                                                (60, 90, "en", 0.98)))
    assert [s.lang for s in spans] == ["en"]

def test_an_unsupported_language_is_treated_as_uncertain():
    # Arabic scores high on Hebrew audio sometimes; only he/en are transcribable.
    spans = scan_language_spans(90.0, _timeline((0, 30, "he", 0.98),
                                                (30, 60, "ar", 0.95),
                                                (60, 90, "he", 0.98)))
    assert [s.lang for s in spans] == ["he"]

def test_a_stray_couple_of_seconds_does_not_become_a_span():
    # Two seconds is detector noise, not a language switch.
    spans = scan_language_spans(90.0, _timeline((0, 44, "en", 0.98),
                                                (44, 46, "he", 0.98),
                                                (46, 90, "en", 0.98)))
    assert [s.lang for s in spans] == ["en"]

def test_a_genuine_short_hebrew_sentence_survives():
    # The counter-case to the guard above: an 8-second Hebrew sentence is real
    # speech and must be transcribed as Hebrew, not translated into English.
    spans = scan_language_spans(90.0, _timeline((0, 41, "en", 0.98),
                                                (41, 49, "he", 0.98),
                                                (49, 90, "en", 0.98)))
    assert [s.lang for s in spans] == ["en", "he", "en"]
    hebrew = [s for s in spans if s.lang == "he"][0]
    assert hebrew.end - hebrew.start >= 5.0

def test_a_file_with_no_confident_detection_falls_back():
    spans = scan_language_spans(60.0, _timeline((0, 60, "ja", 0.30)), fallback="he")
    assert spans == [LangSpan(0.0, 60.0, "he")]

def test_an_empty_recording_yields_a_single_fallback_span():
    spans = scan_language_spans(0.0, _timeline(), fallback="en")
    assert [s.lang for s in spans] == ["en"]

def test_a_file_shorter_than_one_window_is_still_detected():
    spans = scan_language_spans(12.0, _timeline((0, 12, "he", 0.99)))
    assert spans == [LangSpan(0.0, 12.0, "he")]

def test_the_scan_costs_one_probe_per_hop():
    # Each probe is a model call, so this is the cost of the whole feature:
    # duration / hop. At the 2s default a 24-minute meeting is ~720 probes,
    # measured at roughly two minutes with the small detection model.
    calls = []
    base = _timeline((0, 120, "en", 0.98))
    def counting(start, end):
        calls.append((start, end))
        return base(start, end)
    scan_language_spans(120.0, counting)
    assert len(calls) == 60, f"expected 120/2 probes, got {len(calls)}"

def test_probes_look_further_ahead_than_they_label():
    # Each probe reads 6s for context but only labels the 2s at its start.
    windows = []
    def recording(start, end):
        windows.append(round(end - start, 3))
        return "en", 0.98
    scan_language_spans(60.0, recording)
    assert max(windows) > 2.0, "probes must see more context than they label"
