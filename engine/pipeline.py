import tempfile
from pathlib import Path
from engine.audio import normalize_audio, probe_duration
from engine.detect import LangSpan, detect_language_spans
from engine.transcribe import (load_detection_model, load_model, model_ref,
                               transcribe_span, transcribe_wav)
from engine.diarize import diarize_wav
from engine.merge import merge_words_and_turns
from engine.types import TranscriptResult
from engine.config import Config

# Anchor to the project root so the models are found no matter where the
# process was started from.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

def _noop(_stage: str): pass

def transcribe_file(src: Path, config: Config, lang: str | None = None,
                    diarize: bool = True, progress=None,
                    num_speakers: int | None = None) -> TranscriptResult:
    progress = progress or _noop
    models_dir = MODELS_DIR
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "audio.wav"
        progress("normalizing")
        normalize_audio(src, wav)

        spans: list[LangSpan] | None = None
        if lang is None:
            progress("detecting language")
            # One label for the whole file makes Whisper *translate* any span in
            # the other language instead of transcribing it, so find the spans.
            # A small model does this: cheap enough to probe every 2 seconds.
            spans = detect_language_spans(wav, load_detection_model(config.compute_type),
                                          config.fallback_language)
            languages = list(dict.fromkeys(s.lang for s in spans))
            if len(spans) == 1:
                spans = None      # one language throughout: no slicing needed
        else:
            languages = [lang]

        def _model_for(language: str):
            return load_model(language, models_dir, config.compute_type)

        if spans is None:
            # The common case, and the same single pass as before spans existed.
            progress("transcribing")
            words, duration = transcribe_wav(wav, _model_for(languages[0]),
                                             languages[0])
        else:
            duration = probe_duration(wav)
            words = []
            for language in languages:
                progress(f"transcribing {language}")
                model = _model_for(language)
                for span in [s for s in spans if s.lang == language]:
                    words.extend(transcribe_span(wav, model, language,
                                                 span.start, span.end))
            words.sort(key=lambda w: w.start)

        turns = []
        if diarize:
            progress("diarizing")
            turns = diarize_wav(wav, config.hf_token, num_speakers=num_speakers)

        progress("merging")
        segments = merge_words_and_turns(words, turns)

    return TranscriptResult(
        language="+".join(languages),
        model="+".join(model_ref(l, models_dir) for l in languages),
        duration=duration,
        segments=segments,
        words=words,
    )
