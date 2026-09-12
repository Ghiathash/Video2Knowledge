from dataclasses import dataclass


@dataclass
class ReportSection:
    title: str
    start: float
    end: float
    summary: str
    key_points: list[str]
    visual_timestamps: list[float]


@dataclass
class KnowledgeReport:
    title: str
    overview: str
    sections: list[ReportSection]