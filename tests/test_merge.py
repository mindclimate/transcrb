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
