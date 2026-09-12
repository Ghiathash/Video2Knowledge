from dataclasses import dataclass

from src.schemas.transcript import TranscriptSegment


@dataclass
class TranscriptSection:
    start: float
    end: float
    text: str
    source_segments: list[TranscriptSegment]