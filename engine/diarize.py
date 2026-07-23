from pathlib import Path
from engine.types import SpeakerTurn

def _default_factory(hf_token: str):
    from pyannote.audio import Pipeline
    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )

def diarize_wav(wav: Path, hf_token: str | None, pipeline_factory=None) -> list[SpeakerTurn]:
    if not hf_token:
        return []
    factory = pipeline_factory or _default_factory
    try:
        pipeline = factory(hf_token)
        annotation = pipeline(str(wav))
        turns = [
            SpeakerTurn(start=float(turn.start), end=float(turn.end), speaker=speaker)
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        turns.sort(key=lambda t: t.start)
        return turns
    except Exception:
        return []
