"""Gemini adapters around the stable pipeline implementations."""

from contextlib import contextmanager
import os
from pathlib import Path
from threading import RLock


_ENV_LOCK = RLock()


@contextmanager
def _configured(api_key: str | None):
    """Expose a session-only key to legacy Gemini modules for one guarded call."""
    if not api_key:
        yield
        return
    with _ENV_LOCK:
        previous = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = api_key
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = previous


class GeminiVisionProvider:
    name = "Gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model, self.api_key = model or "configured Gemini model", api_key

    def analyze(self, image_path: Path, timestamp: float, transcript_context: str):
        from src.visuals import vlm_analyzer
        if self.model != "configured Gemini model":
            vlm_analyzer.GEMINI_MODEL = self.model
        with _configured(self.api_key):
            return vlm_analyzer.analyze_visual(image_path, timestamp, transcript_context)


class GeminiLLMProvider:
    name = "Gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model, self.api_key = model or "configured Gemini model", api_key

    def synthesize(self, aligned_sections_path: str, checkpoint_path: str):
        from src.synthesis import synthesizer
        if self.model != "configured Gemini model":
            synthesizer.GEMINI_MODEL = self.model
        with _configured(self.api_key):
            return synthesizer.synthesize_report(aligned_sections_path, checkpoint_path=checkpoint_path)


class GeminiVerificationProvider:
    name = "Gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model, self.api_key = model or "configured Gemini model", api_key

    def verify(self, aligned_sections_path: str, draft_report_path: str, checkpoint_path: str):
        from src.verification import faithfulness
        if self.model != "configured Gemini model":
            faithfulness.GEMINI_MODEL = self.model
        with _configured(self.api_key):
            return faithfulness.verify_report(aligned_sections_path, draft_report_path, checkpoint_path)
