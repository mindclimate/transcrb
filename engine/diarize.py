import logging
from pathlib import Path
from engine.types import SpeakerTurn

log = logging.getLogger(__name__)

def _default_factory(hf_token: str):
    from pyannote.audio import Pipeline
    try:
        return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
    except TypeError:
        return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)

def _devices() -> list[str]:
    """Devices to attempt, best first.

    Diarization was the single most expensive stage in the pipeline while it ran
    on the CPU: 119.5s against 15.1s on Metal for the same two minutes of audio,
    roughly half of the total runtime. The accelerator therefore goes first.

    A CPU entry always comes last. pyannote reaches torch ops that MPS does not
    implement, and when it does the cost has to be speed rather than the speaker
    labels themselves.
    """
    try:
        import torch
    except ImportError:            # torch is a hard dependency; be safe anyway
        return ["cpu"]
    if torch.backends.mps.is_available():
        return ["mps", "cpu"]
    if torch.cuda.is_available():
        return ["cuda", "cpu"]
    return ["cpu"]

def _on_device(pipeline, device: str):
    """Move `pipeline` to `device`, tolerating anything without pyannote's .to()."""
    to = getattr(pipeline, "to", None)
    if to is None:
        return pipeline
    import torch
    # pyannote's .to() returns self; a stub that returns None must not erase it.
    return to(torch.device(device)) or pipeline

def _to_turns(result) -> list[SpeakerTurn]:
    """Read speaker turns from either pyannote API generation.

    4.x pipelines return a DiarizeOutput exposing serialize(); 3.x returned an
    Annotation with itertracks(). Supporting both means a pyannote upgrade
    cannot silently cost us speaker labels.
    """
    if hasattr(result, "serialize"):
        rows = result.serialize().get("diarization", [])
        return [
            SpeakerTurn(start=float(r["start"]), end=float(r["end"]), speaker=r["speaker"])
            for r in rows
        ]
    if hasattr(result, "itertracks"):
        return [
            SpeakerTurn(start=float(turn.start), end=float(turn.end), speaker=speaker)
            for turn, _, speaker in result.itertracks(yield_label=True)
        ]
    raise TypeError(f"unrecognised diarization result type: {type(result).__name__}")

def diarize_wav(wav: Path, hf_token: str | None, pipeline_factory=None,
                num_speakers: int | None = None) -> list[SpeakerTurn]:
    """Speaker turns for `wav`.

    `num_speakers` pins how many people to expect. Left unset, clustering
    decides — which can over-split when a voice shifts in pitch or tone.
    """
    if not hf_token:
        log.warning("No HuggingFace token — skipping diarization; transcript will have one speaker.")
        return []
    factory = pipeline_factory or _default_factory
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    devices = _devices()
    for i, device in enumerate(devices):
        last = i == len(devices) - 1
        try:
            # Rebuilt per attempt: a pipeline that failed partway through being
            # moved to the GPU is not a safe thing to retry on the CPU.
            pipeline = _on_device(factory(hf_token), device)
            turns = _to_turns(pipeline(str(wav), **kwargs))
            turns.sort(key=lambda t: t.start)
            if not turns:
                log.warning("Diarization produced no speaker turns.")
            return turns
        except Exception:
            if not last:
                log.warning("Diarization on %s failed; retrying on %s.",
                            device, devices[i + 1], exc_info=True)
                continue
            # Never fail the transcript over diarization — but say so loudly, or
            # a silent failure is indistinguishable from a genuine single speaker.
            log.exception("Diarization failed; continuing without speaker labels.")
            return []
    return []
