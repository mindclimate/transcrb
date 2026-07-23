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
