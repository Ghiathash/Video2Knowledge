from dataclasses import dataclass
from pathlib import Path


@dataclass
class FrameInfo:
    path: Path
    timestamp: float