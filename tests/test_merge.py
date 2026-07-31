from engine.types import Word, SpeakerTurn
from engine.merge import merge_words_and_turns

def test_basic_two_speakers():
    words = [Word(0.0, 1.0, "Hello"), Word(1.0, 2.0, "there"), Word(2.5, 3.5, "Hi")]
    turns = [SpeakerTurn(0.0, 2.2, "SPEAKER_00"), SpeakerTurn(2.3, 4.0, "SPEAKER_01")]
    segs = merge_words_and_turns(words, turns)
    assert len(segs) == 2
    assert segs[0].speaker == "SPEAKER_00"
    assert segs[0].text == "Hello there"
    assert segs[1].speaker == "SPEAKER_01"
    assert segs[1].text == "Hi"
    assert segs[0].start == 0.0 and segs[0].end == 2.0

def test_word_spanning_boundary_goes_to_majority_overlap():
    words = [Word(2.0, 2.4, "edge")]
    turns = [SpeakerTurn(0.0, 2.2, "SPEAKER_00"), SpeakerTurn(2.2, 4.0, "SPEAKER_01")]
    segs = merge_words_and_turns(words, turns)
    # overlap with SPEAKER_00 = 0.2, with SPEAKER_01 = 0.2 -> tie -> earlier turn wins
    assert segs[0].speaker == "SPEAKER_00"

def test_no_turns_returns_single_speaker():
    words = [Word(0.0, 1.0, "solo"), Word(1.0, 2.0, "run")]
    segs = merge_words_and_turns(words, [])
    assert len(segs) == 1
    assert segs[0].speaker == "SPEAKER_00"
    assert segs[0].text == "solo run"

def test_empty_words_returns_empty():
    assert merge_words_and_turns([], [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]) == []

def test_long_pause_splits_same_speaker():
    # A 2s silence reads as a paragraph break even with one speaker.
    words = [Word(0.0, 1.0, "before"), Word(3.0, 4.0, "after")]
    segs = merge_words_and_turns(words, [])
    assert len(segs) == 2
    assert segs[0].text == "before"
    assert segs[1].text == "after"

def test_short_pause_does_not_split():
    words = [Word(0.0, 1.0, "same"), Word(1.2, 2.0, "breath")]
    segs = merge_words_and_turns(words, [])
    assert len(segs) == 1
    assert segs[0].text == "same breath"

def test_long_monologue_splits_at_sentence_end():
    # 40s of continuous speech must not become one wall of text; it breaks
    # past MAX_SEGMENT_SECONDS, but only at a sentence boundary.
    words = [Word(float(i), float(i) + 0.9, ("word." if i % 10 == 9 else "word"))
             for i in range(40)]
    segs = merge_words_and_turns(words, [])
    assert len(segs) > 1
    # every split lands right after a sentence-ending word
    for seg in segs[:-1]:
        assert seg.text.endswith(".")

def test_long_monologue_without_sentence_ends_stays_whole():
    # No punctuation to split on -> we keep it intact rather than cut a clause.
    words = [Word(float(i), float(i) + 0.9, "word") for i in range(40)]
    segs = merge_words_and_turns(words, [])
    assert len(segs) == 1


# ---- mixed-language recordings ----------------------------------------------
# Words from either side of a language switch must not land in one segment: the
# line would mix scripts and read as gibberish in both directions.

def test_segments_break_when_the_language_changes():
    words = [Word(0.0, 1.0, "hello", lang="en"),
             Word(1.1, 2.0, "everyone", lang="en"),
             Word(2.1, 3.0, "שלום", lang="he"),
             Word(3.1, 4.0, "לכולם", lang="he")]
    segs = merge_words_and_turns(words, [])
    assert len(segs) == 2                      # not a pause or speaker break
    assert segs[0].text == "hello everyone"
    assert segs[1].text == "שלום לכולם"

def test_each_segment_carries_the_language_of_its_words():
    words = [Word(0.0, 1.0, "hello", lang="en"), Word(2.1, 3.0, "שלום", lang="he")]
    segs = merge_words_and_turns(words, [])
    assert [s.lang for s in segs] == ["en", "he"]

def test_switching_back_produces_three_segments():
    words = [Word(0.0, 1.0, "hi", lang="en"),
             Word(1.1, 2.0, "שלום", lang="he"),
             Word(2.1, 3.0, "bye", lang="en")]
    assert [s.lang for s in merge_words_and_turns(words, [])] == ["en", "he", "en"]

def test_single_language_words_still_merge_into_one_segment():
    words = [Word(0.0, 1.0, "one", lang="en"), Word(1.1, 2.0, "two", lang="en")]
    segs = merge_words_and_turns(words, [])
    assert len(segs) == 1 and segs[0].lang == "en"


# ---- speaker assignment on long recordings -----------------------------------
# Comparing every word against every turn is quadratic: a three-hour meeting is
# ~36k words against ~3.3k turns, so the scan alone runs into minutes. The
# bucketed lookup must give *identical* labels — a faster merge that reassigns
# speech to the wrong person is the failure this project exists to prevent.

import random
import time as _time
from engine.merge import _assign_speaker, _overlap

def _assign_by_linear_scan(word, turns):
    """The original implementation, kept as the oracle."""
    best_speaker, best_overlap = "SPEAKER_00", -1.0
    for t in turns:
        ov = _overlap(word.start, word.end, t.start, t.end)
        if ov > best_overlap:
            best_overlap, best_speaker = ov, t.speaker
    return best_speaker

def test_assignment_matches_the_linear_scan_on_random_recordings():
    rng = random.Random(20260730)
    for _ in range(40):
        turns, t = [], 0.0
        for i in range(rng.randint(1, 60)):
            start = t + rng.uniform(-0.3, 2.0)      # gaps and overlaps both occur
            end = start + rng.uniform(0.05, 8.0)
            turns.append(SpeakerTurn(start, end, f"SPEAKER_{i % 4:02d}"))
            t = end
        span = max(x.end for x in turns) + 3.0
        for _ in range(200):
            ws = rng.uniform(-1.0, span)
            word = Word(ws, ws + rng.uniform(0.01, 1.5), "w")
            assert _assign_speaker(word, turns) == _assign_by_linear_scan(word, turns)

def test_a_word_in_silence_keeps_the_historical_fallback():
    # No turn overlaps it at all. The original picked the first turn in the list
    # (its "best" started below zero overlap), and downstream output depends on
    # that, so the behaviour is pinned rather than quietly improved.
    turns = [SpeakerTurn(10.0, 11.0, "SPEAKER_01"), SpeakerTurn(12.0, 13.0, "SPEAKER_02")]
    assert _assign_speaker(Word(0.0, 0.5, "quiet"), turns) == "SPEAKER_01"

def test_unsorted_turns_are_handled():
    turns = [SpeakerTurn(5.0, 6.0, "SPEAKER_01"), SpeakerTurn(0.0, 1.0, "SPEAKER_00")]
    assert _assign_speaker(Word(0.2, 0.8, "early"), turns) == "SPEAKER_00"
    assert _assign_speaker(Word(5.2, 5.8, "late"), turns) == "SPEAKER_01"

def test_a_long_recording_merges_quickly():
    # 3 hours: ~36k words against ~3.3k turns. The quadratic scan needed over a
    # minute here; the bound is deliberately loose, it only has to catch a
    # regression back to quadratic.
    turns = [SpeakerTurn(i * 3.2, i * 3.2 + 3.0, f"SPEAKER_{i % 3:02d}")
             for i in range(3300)]
    words = [Word(i * 0.3, i * 0.3 + 0.25, "word") for i in range(36000)]
    start = _time.perf_counter()
    segs = merge_words_and_turns(words, turns)
    assert _time.perf_counter() - start < 10.0
    assert segs
