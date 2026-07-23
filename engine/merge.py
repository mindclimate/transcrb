from engine.types import Word, SpeakerTurn, Segment

# A segment ends when the speaker changes, but also when a single speaker runs
# on: a long pause reads as a paragraph break, and past a maximum length the
# text becomes an unreadable wall regardless of pauses.
PAUSE_BREAK_SECONDS = 1.5
MAX_SEGMENT_SECONDS = 30.0
SENTENCE_ENDINGS = (".", "?", "!", "。", "؟")

def _overlap(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def _assign_speaker(word: Word, turns: list[SpeakerTurn]) -> str:
    best_speaker = "SPEAKER_00"
    best_overlap = -1.0
    for t in turns:
        ov = _overlap(word.start, word.end, t.start, t.end)
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = t.speaker
    return best_speaker

def _should_break(cur_words: list[Word], word: Word) -> bool:
    """True when the run so far should be closed before adding `word`."""
    if not cur_words:
        return False
    gap = word.start - cur_words[-1].end
    if gap >= PAUSE_BREAK_SECONDS:
        return True
    if word.end - cur_words[0].start >= MAX_SEGMENT_SECONDS:
        # Only split mid-flow at a sentence end, so we don't cut a clause.
        return cur_words[-1].text.strip().endswith(SENTENCE_ENDINGS)
    return False

def merge_words_and_turns(words: list[Word], turns: list[SpeakerTurn]) -> list[Segment]:
    if not words:
        return []
    segments: list[Segment] = []
    cur_speaker = None
    cur_words: list[Word] = []
    for w in words:
        speaker = _assign_speaker(w, turns) if turns else "SPEAKER_00"
        if cur_words and (speaker != cur_speaker or _should_break(cur_words, w)):
            segments.append(_flush(cur_speaker, cur_words))
            cur_words = []
        cur_speaker = speaker
        cur_words.append(w)
    if cur_words:
        segments.append(_flush(cur_speaker, cur_words))
    return segments

def _flush(speaker: str | None, words: list[Word]) -> Segment:
    text = " ".join(w.text.strip() for w in words).strip()
    return Segment(start=words[0].start, end=words[-1].end,
                   speaker=speaker or "SPEAKER_00", text=text)
