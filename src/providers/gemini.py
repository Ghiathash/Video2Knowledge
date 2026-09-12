from pathlib import Path


class GeminiVisionProvider:
    name = "Gemini API"

    def __init__(self, model: str | None = None):
        self.model = model or "configured Gemini model"

    def analyze(self, image_path: Path, timestamp: float, transcript_context: str):
        from src.visuals import vlm_analyzer
        if self.model != "configured Gemini model":
            vlm_analyzer.GEMINI_MODEL = self.model
        return vlm_analyzer.analyze_visual(image_path, timestamp, transcript_context)


class GeminiLLMProvider:
    name = "Gemini API"

    def __init__(self, model: str | None = None):
        self.model = model or "configured Gemini model"

    def synthesize(self, aligned_sections_path: str, checkpoint_path: str):
        from src.synthesis import synthesizer
        if self.model != "configured Gemini model":
            synthesizer.GEMINI_MODEL = self.model
        return synthesizer.synthesize_report(aligned_sections_path, checkpoint_path=checkpoint_path)


class GeminiVerificationProvider:
    name = "Gemini API"

    def __init__(self, model: str | None = None):
        self.model = model or "configured Gemini model"

    def verify(self, aligned_sections_path: str, draft_report_path: str, checkpoint_path: str):
        from src.verification import faithfulness
        if self.model != "configured Gemini model":
            faithfulness.GEMINI_MODEL = self.model
        return faithfulness.verify_report(aligned_sections_path, draft_report_path, checkpoint_path)
