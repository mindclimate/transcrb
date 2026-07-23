import tempfile
from pathlib import Path
from engine.audio import normalize_audio, probe_duration
from engine.detect import detect_language
from engine.transcribe import load_model, transcribe_wav, model_ref
from engine.diarize import diarize_wav
from engine.merge import merge_words_and_turns
from engine.types import TranscriptResult
from engine.config import Config

def _noop(_stage: str): pass

def transcribe_file(src: Path, config: Config, lang: str | None = None,
                    diarize: bool = True, progress=None,
                    num_speakers: int | None = None) -> TranscriptResult:
    progress = progress or _noop
    models_dir = Path("models")
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "audio.wav"
        progress("normalizing")
        normalize_audio(src, wav)

        det_model = None
        if lang is None:
            progress("detecting language")
            det_model = load_model("en", models_dir, config.compute_type)  # large-v3 is multilingual
            lang, _prob = detect_language(wav, det_model, config.fallback_language)

        progress("transcribing")
        # Detection already loaded large-v3; reloading it for an English file
        # would cost a second multi-GB load for the same weights.
        if det_model is not None and model_ref(lang, models_dir) == model_ref("en", models_dir):
            model = det_model
        else:
            model = load_model(lang, models_dir, config.compute_type)
        words, duration = transcribe_wav(wav, model, lang)

        turns = []
        if diarize:
            progress("diarizing")
            turns = diarize_wav(wav, config.hf_token, num_speakers=num_speakers)

        progress("merging")
        segments = merge_words_and_turns(words, turns)

    return TranscriptResult(
        language=lang,
        model=model_ref(lang, models_dir),
        duration=duration,
        segments=segments,
    )
