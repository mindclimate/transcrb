from pathlib import Path
from faster_whisper import WhisperModel
from engine.types import Word

EN_MODEL_ID = "large-v3"
HE_MODEL_DIRNAME = "ivrit-whisper-ct2"

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
            words.append(Word(start=w.start, end=w.end, text=w.word))
    return words, float(info.duration)
