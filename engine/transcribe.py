from pathlib import Path

from faster_whisper import WhisperModel

from engine.types import Word

# Turbo keeps all 32 encoder layers but has 4 decoder layers instead of 32, and
# on this CPU that is worth more than moving large-v3 to the GPU: 28.2s against
# 113.9s for two minutes of audio (4.0x), where the same weights on Metal via MLX
# managed only 34.4s. Same runtime, same API — the win is purely architectural.
EN_MODEL_ID = "large-v3-turbo"
HE_MODEL_DIRNAME = "ivrit-whisper-turbo-ct2"

# The pre-turbo download location. An install that has not re-run setup still has
# working Hebrew weights here, and should keep transcribing on them rather than
# failing on a directory that does not exist yet.
HE_LEGACY_DIRNAME = "ivrit-whisper-ct2"

# Language identification only — never used to produce text. `base` is ~20x
# cheaper per probe than large-v3 and measured *more* accurate at spotting a
# switch, which is what makes scanning a long meeting for language spans
# affordable at a 2-second resolution.
DETECT_MODEL_ID = "base"

def load_detection_model(compute_type: str = "int8") -> WhisperModel:
    return WhisperModel(DETECT_MODEL_ID, device="cpu", compute_type=compute_type)

def model_ref(language: str, models_dir: Path) -> str:
    if language == "he":
        turbo = Path(models_dir) / HE_MODEL_DIRNAME
        legacy = Path(models_dir) / HE_LEGACY_DIRNAME
        if not turbo.exists() and legacy.exists():
            return str(legacy)
        return str(turbo)
    return EN_MODEL_ID

def load_model(language: str, models_dir: Path, compute_type: str) -> WhisperModel:
    return WhisperModel(model_ref(language, models_dir), device="cpu",
                        compute_type=compute_type)

def transcribe_wav(audio, model, language: str) -> tuple[list[Word], float]:
    """Transcribe a whole recording in one language.

    Takes the samples the caller already read, so the audio is not decoded a
    second time inside the model.
    """
    segments, info = model.transcribe(audio, language=language,
                                      word_timestamps=True)
    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            words.append(Word(start=w.start, end=w.end, text=w.word,
                              lang=language))
    return words, float(info.duration)

def transcribe_span(audio, rate: int, model, language: str, start: float,
                    end: float) -> list[Word]:
    """Transcribe one language span, returning words on the file's timeline.

    The span is sliced out and handed to the model as samples rather than using
    ffmpeg-style clip arguments: the returned timestamps are then unambiguously
    relative to the slice, and shifting them by `start` is exact. Getting this
    wrong would misalign every word against the speaker turns.

    Takes samples rather than a path so the caller reads the recording once,
    rather than once per span.
    """
    chunk = audio[int(start * rate):int(end * rate)]
    segments, _info = model.transcribe(chunk, language=language,
                                       word_timestamps=True)
    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            words.append(Word(start=start + w.start, end=start + w.end,
                              text=w.word, lang=language))
    return words
