from dataclasses import dataclass, field

@dataclass
class Word:
    start: float
    end: float
    text: str

@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str

@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str

@dataclass
class TranscriptResult:
    language: str
    model: str
    duration: float
    segments: list = field(default_factory=list)
    # Raw word timings are kept so a transcript can be re-segmented later
    # without paying for another transcription pass.
    words: list = field(default_factory=list)
