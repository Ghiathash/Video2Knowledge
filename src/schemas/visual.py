from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CandidateFrame:
    path: Path
    timestamp: float
    change_score: float = 0.0
    sources: list[str] = field(default_factory=list)
    evidence: str | None = None


@dataclass
class RankedVisual:
    path: Path
    timestamp: float
    change_score: float
    visual_richness_score: float
    importance_score: float


@dataclass
class VisualAnalysis:
    path: Path
    timestamp: float

    visual_type: str
    contains_knowledge: bool
    importance_score: float

    description: str
    visible_text: str

    knowledge_points: list[str]
    transcript_alignment: str