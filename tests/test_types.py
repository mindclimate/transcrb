from engine.types import Word, SpeakerTurn, Segment, TranscriptResult

def test_transcript_result_holds_segments():
    seg = Segment(start=0.0, end=2.0, speaker="SPEAKER_00", text="hi")
    res = TranscriptResult(language="en", model="large-v3", duration=2.0, segments=[seg])
    assert res.segments[0].text == "hi"
    assert res.language == "en"

def test_word_and_turn_fields():
    w = Word(start=0.0, end=1.0, text="hello")
    t = SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_01")
    assert w.text == "hello" and t.speaker == "SPEAKER_01"
