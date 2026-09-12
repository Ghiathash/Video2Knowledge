from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoInfo:
    path: Path
    duration: float
    fps: float
    width: int
    height: int
    video_codec: str
    has_audio: bool
    audio_codec: str | None = None