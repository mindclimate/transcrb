from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel

from engine.types import Word

EN_MODEL_ID = "large-v3"
HE_MODEL_DIRNAME = "ivrit-whisper-ct2"

# Language identification only — never used to produce text. `base` is ~20x
# cheaper per probe than large-v3 and measured *more* accurate at spotting a
# switch, which is what makes scanning a long meeting for language spans
# affordable at a 2-second resolution.
DETECT_MODEL_ID = "base"

def load_detection_model(compute_type: str = "int8") -> WhisperModel:
    return WhisperModel(DETECT_MODEL_ID, device="cpu", compute_type=compute_type)

def model_ref(language: str, models_dir: Path) -> str:
    if language == "he":
        return str(Path(models_dir) / HE_MODEL_DIRNAME)
    return EN_MODEL_ID

def load_model(language: str, models_dir: Path, compute_type: str) -> WhisperModel:
    return WhisperModel(model_ref(language, models_dir), device="cpu",
                        compute_type=compute_type)

def transcribe_wav(wav: Path, model, language: str) -> tuple[list[Word], float]:
    segments, info = model.transcribe(str(wav), language=language,
                                      word_timestamps=True)
    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            words.append(Word(start=w.start, end=w.end, text=w.word,
                              lang=language))
    return words, float(info.duration)

def transcribe_span(wav: Path, model, language: str, start: float,
                    end: float) -> list[Word]:
    """Transcribe one language span, returning words on the file's timeline.

    The span is sliced out and handed to the model as samples rather than using
    ffmpeg-style clip arguments: the returned timestamps are then unambiguously
    relative to the slice, and shifting them by `start` is exact. Getting this
    wrong would misalign every word against the speaker turns.
    """
    audio, rate = sf.read(str(wav), dtype="float32")
    chunk = audio[int(start * rate):int(end * rate)]
    segments, _info = model.transcribe(chunk, language=language,
                                       word_timestamps=True)
    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            words.append(Word(start=start + w.start, end=start + w.end,
                              text=w.word, lang=language))
    return words
