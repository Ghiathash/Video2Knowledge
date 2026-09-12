"""Local Ollama providers. Models must be installed explicitly by the user."""

import base64
import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from src.schemas.report import KnowledgeReport, ReportSection
from src.schemas.visual import VisualAnalysis


class OllamaError(RuntimeError):
    pass


def installed_ollama_models(base_url: str | None = None) -> set[str]:
    endpoint = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
    try:
        with urlopen(f"{endpoint}/api/tags", timeout=3) as response:
            data = json.load(response)
        return {item["name"] for item in data.get("models", []) if item.get("name")}
    except (OSError, URLError, json.JSONDecodeError, KeyError, TypeError):
        return set()


class _OllamaClient:
    def __init__(self, model: str, base_url: str | None = None):
        if not model:
            raise ValueError("A local Ollama model must be configured.")
        self.model = model
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")

    def generate_json(self, prompt: str, images: list[str] | None = None) -> dict:
        payload = {"model": self.model, "prompt": prompt, "stream": False, "format": "json"}
        if images:
            payload["images"] = images
        request = Request(
            f"{self.base_url}/api/generate", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=600) as response:
                body = json.load(response)
        except (OSError, URLError, json.JSONDecodeError) as error:
            raise OllamaError(
                "Local Ollama is unavailable. Start Ollama and install the configured model."
            ) from error
        try:
            return json.loads(body["response"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise OllamaError("The local model did not return valid JSON.") from error


class OllamaVisionProvider:
    name = "Local Ollama"

    def __init__(self, model: str):
        self.model = model
        self.client = _OllamaClient(model)

    def analyze(self, image_path: Path, timestamp: float, transcript_context: str) -> VisualAnalysis:
        prompt = (
            "Analyze this educational frame using the nearby transcript. Return JSON with keys "
            "visual_type, contains_knowledge, importance_score (0 to 1), description, "
            "visible_text, knowledge_points (list), transcript_alignment. Do not invent visible text.\n"
            f"Transcript: {transcript_context}"
        )
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        data = self.client.generate_json(prompt, [encoded])
        return VisualAnalysis(path=image_path, timestamp=timestamp, **data)


class OllamaLLMProvider:
    name = "Local Ollama"

    def __init__(self, model: str):
        self.model = model
        self.client = _OllamaClient(model)

    def synthesize(self, aligned_sections_path: str, checkpoint_path: str) -> KnowledgeReport:
        sections = json.loads(Path(aligned_sections_path).read_text(encoding="utf-8"))
        report_sections = []
        for section in sections:
            prompt = (
                "Create a grounded study-report section using only this evidence. Return JSON with "
                "title, summary, key_points. Do not introduce facts or numbers. Evidence:\n"
                + json.dumps(section, ensure_ascii=False)
            )
            data = self.client.generate_json(prompt)
            report_sections.append(ReportSection(
                title=data["title"], start=float(section["start"]), end=float(section["end"]),
                summary=data["summary"], key_points=data["key_points"],
                visual_timestamps=[float(v["timestamp"]) for v in section["visuals"]],
            ))
        overview = self.client.generate_json(
            "Using only these report sections, return JSON with title and overview:\n"
            + json.dumps([vars(item) for item in report_sections], ensure_ascii=False)
        )
        return KnowledgeReport(overview["title"], overview["overview"], report_sections)


class OllamaVerificationProvider:
    name = "Local Ollama"

    def __init__(self, model: str):
        self.model = model
        self.client = _OllamaClient(model)

    def verify(self, aligned_sections_path: str, draft_report_path: str, checkpoint_path: str):
        from src.synthesis.synthesizer import _validate_numeric_grounding
        source = json.loads(Path(aligned_sections_path).read_text(encoding="utf-8"))
        draft = json.loads(Path(draft_report_path).read_text(encoding="utf-8"))
        verified, audit = [], []
        for evidence, section in zip(source, draft["sections"], strict=True):
            prompt = (
                "Verify this draft strictly against the evidence. Return JSON with title, summary, "
                "key_points, is_faithful, issues. Preserve supported meaning and numbers.\nEvidence:\n"
                + json.dumps(evidence, ensure_ascii=False) + "\nDraft:\n" + json.dumps(section, ensure_ascii=False)
            )
            data = self.client.generate_json(prompt)
            trusted = evidence["text"] + "\n" + "\n".join(v.get("visible_text", "") for v in evidence["visuals"])
            generated = "\n".join([data["title"], data["summary"], *data["key_points"]])
            try:
                _validate_numeric_grounding(generated, trusted, "Local verification")
                selected = data
            except ValueError as error:
                selected = section
                data["is_faithful"] = False
                data["issues"] = [f"Verifier rewrite rejected: {error}"]
            verified.append(ReportSection(
                title=selected["title"], start=float(evidence["start"]), end=float(evidence["end"]),
                summary=selected["summary"], key_points=selected["key_points"],
                visual_timestamps=[float(v["timestamp"]) for v in evidence["visuals"]],
            ))
            audit.append({"section_index": evidence["section_index"], "is_faithful": data["is_faithful"], "issues": data["issues"]})
        return KnowledgeReport(draft["title"], draft["overview"], verified), audit
