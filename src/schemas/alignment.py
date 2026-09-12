from dataclasses import dataclass

from src.schemas.visual import VisualAnalysis


@dataclass
class AlignedVisual:
    visual: VisualAnalysis
    alignment_score: float
    alignment_method: str


@dataclass
class AlignedSection:
    section_index: int
    start: float
    end: float
    text: str
    visuals: list[AlignedVisual]