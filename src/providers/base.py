from pathlib import Path
from typing import Protocol

from src.schemas.report import KnowledgeReport
from src.schemas.transcript import TranscriptResult, TranscriptSegment
from src.schemas.visual import VisualAnalysis


class ASRProvider(Protocol):
    name: str
    model: str

    def transcribe(self, audio_path: Path, *, language: str | None) -> TranscriptResult: ...


class VisionProvider(Protocol):
    name: str
    model: str

    def analyze(self, image_path: Path, timestamp: float, transcript_context: str) -> VisualAnalysis: ...


class LLMProvider(Protocol):
    name: str
    model: str

    def synthesize(self, aligned_sections_path: str, checkpoint_path: str) -> KnowledgeReport: ...


class VerificationProvider(Protocol):
    name: str
    model: str

    def verify(self, aligned_sections_path: str, draft_report_path: str, checkpoint_path: str): ...
