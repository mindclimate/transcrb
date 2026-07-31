from engine.types import Word, SpeakerTurn, Segment

# A segment ends when the speaker changes, but also when a single speaker runs
# on: a long pause reads as a paragraph break, and past a maximum length the
# text becomes an unreadable wall regardless of pauses.
PAUSE_BREAK_SECONDS = 1.5
MAX_SEGMENT_SECONDS = 30.0
SENTENCE_ENDINGS = (".", "?", "!", "。", "؟")

def _overlap(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

# Comparing every word against every turn is quadratic, and meetings are long: a
# three-hour recording is ~36k words against ~3.3k turns, where the scan alone
# ran over a minute. Turns are bucketed by time so each word only looks at the
# handful that could possibly overlap it.
_BUCKET_SECONDS = 5.0

def _buckets(start: float, end: float) -> range:
    return range(int(start // _BUCKET_SECONDS), int(end // _BUCKET_SECONDS) + 1)

def _bucket_index(turns: list[SpeakerTurn]) -> dict[int, list[int]]:
    """Time slot -> indices of the turns touching it, ascending.

    Any two intervals that overlap by a positive amount share at least one slot,
    so a turn absent from a word's slots cannot have beaten the winner.
    """
    index: dict[int, list[int]] = {}
    for i, t in enumerate(turns):
        for b in _buckets(t.start, t.end):
            index.setdefault(b, []).append(i)
    return index

def _assign_speaker(word: Word, turns: list[SpeakerTurn],
                    index: dict[int, list[int]] | None = None) -> str:
    if not turns:
        return "SPEAKER_00"
    index = _bucket_index(turns) if index is None else index
    candidates: set[int] = set()
    for b in _buckets(word.start, word.end):
        candidates.update(index.get(b, ()))
    best_speaker, best_overlap = None, 0.0
    # Ascending index order so an exact tie goes to the earlier turn, which is
    # what the original full scan did.
    for i in sorted(candidates):
        t = turns[i]
        ov = _overlap(word.start, word.end, t.start, t.end)
        if ov > best_overlap:
            best_overlap, best_speaker = ov, t.speaker
    # A word sitting in silence overlaps nothing. The original scan started its
    # best at below-zero overlap and so returned the first turn in the list;
    # transcripts already carry that attribution, so it is preserved.
    return turns[0].speaker if best_speaker is None else best_speaker

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
    index = _bucket_index(turns) if turns else None
    for w in words:
        speaker = _assign_speaker(w, turns, index) if turns else "SPEAKER_00"
        # A language switch always ends the segment: one line must not mix
        # scripts, which would read as gibberish in both directions.
        language_changed = bool(cur_words) and w.lang != cur_words[-1].lang
        if cur_words and (speaker != cur_speaker or language_changed
                          or _should_break(cur_words, w)):
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
                   speaker=speaker or "SPEAKER_00", text=text,
                   lang=words[0].lang)
