from engine.types import Word, SpeakerTurn, Segment

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

def merge_words_and_turns(words: list[Word], turns: list[SpeakerTurn]) -> list[Segment]:
    if not words:
        return []
    segments: list[Segment] = []
    cur_speaker = None
    cur_words: list[Word] = []
    for w in words:
        speaker = _assign_speaker(w, turns) if turns else "SPEAKER_00"
        if speaker != cur_speaker and cur_words:
            segments.append(_flush(cur_speaker, cur_words))
            cur_words = []
        cur_speaker = speaker
        cur_words.append(w)
    if cur_words:
        segments.append(_flush(cur_speaker, cur_words))
    return segments

def _flush(speaker: str, words: list[Word]) -> Segment:
    text = " ".join(w.text.strip() for w in words).strip()
    return Segment(start=words[0].start, end=words[-1].end, speaker=speaker, text=text)
