from pathlib import Path

from src.schemas.report import KnowledgeReport
from src.schemas.transcript import TranscriptResult


class FakeASRProvider:
    name = "Fake ASR"
    model = "fake"

    def transcribe(self, audio_path: Path, *, language: str | None):
        return TranscriptResult(language or "en", 1.0, [])


class FakeVisionProvider:
    name = "Fake Vision"
    model = "fake"


class FakeLLMProvider:
    name = "Fake LLM"
    model = "fake"

    def synthesize(self, aligned_sections_path: str, checkpoint_path: str):
        return KnowledgeReport("Test", "Test", [])
