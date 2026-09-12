"""Generic Chat Completions providers for OpenAI-compatible endpoints."""

import base64
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.schemas.report import KnowledgeReport, ReportSection
from src.schemas.visual import VisualAnalysis


class OpenAICompatibleError(RuntimeError):
    pass


def _json_content(value: str) -> dict:
    text = value.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:].lstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise OpenAICompatibleError("The selected model did not return valid JSON.") from error


class _ChatClient:
    def __init__(self, model: str, base_url: str, api_key: str | None = None):
        if not model or not base_url:
            raise ValueError("Model ID and base URL are required for an OpenAI-compatible model.")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def generate_json(self, prompt: str, image_path: Path | None = None) -> dict:
        content: str | list[dict] = prompt
        if image_path:
            mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}]
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}], "temperature": 0}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(f"{self.base_url}/chat/completions", data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urlopen(request, timeout=600) as response:
                body = json.load(response)
            return _json_content(body["choices"][0]["message"]["content"])
        except (HTTPError, URLError, OSError, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise OpenAICompatibleError("The selected compatible model is unavailable. Check its model ID, base URL, and credentials.") from error


def test_connection(base_url: str, api_key: str | None = None) -> tuple[bool, str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with urlopen(Request(f"{base_url.rstrip('/')}/models", headers=headers), timeout=8) as response:
            json.load(response)
        return True, "Endpoint is reachable."
    except Exception:
        return False, "Could not reach the endpoint. Check the URL and credentials."


class OpenAICompatibleVisionProvider:
    name = "OpenAI-compatible"

    def __init__(self, model: str, base_url: str, api_key: str | None = None):
        self.model, self.client = model, _ChatClient(model, base_url, api_key)

    def analyze(self, image_path: Path, timestamp: float, transcript_context: str) -> VisualAnalysis:
        prompt = ("Analyze this educational frame using the transcript. Return only JSON with keys visual_type, contains_knowledge, importance_score, description, visible_text, knowledge_points, transcript_alignment. Never invent visible text.\nTranscript:\n" + transcript_context)
        return VisualAnalysis(path=image_path, timestamp=timestamp, **self.client.generate_json(prompt, image_path))


class OpenAICompatibleLLMProvider:
    name = "OpenAI-compatible"

    def __init__(self, model: str, base_url: str, api_key: str | None = None):
        self.model, self.client = model, _ChatClient(model, base_url, api_key)

    def synthesize(self, aligned_sections_path: str, checkpoint_path: str) -> KnowledgeReport:
        sections = json.loads(Path(aligned_sections_path).read_text(encoding="utf-8"))
        report_sections = []
        for section in sections:
            data = self.client.generate_json("Create a grounded report section from only this evidence. Return only JSON with title, summary, key_points.\n" + json.dumps(section, ensure_ascii=False))
            report_sections.append(ReportSection(title=data["title"], start=float(section["start"]), end=float(section["end"]), summary=data["summary"], key_points=data["key_points"], visual_timestamps=[float(v["timestamp"]) for v in section["visuals"]]))
        overview = self.client.generate_json("Return only JSON with title and overview for these grounded sections:\n" + json.dumps([vars(item) for item in report_sections], ensure_ascii=False, default=str))
        return KnowledgeReport(overview["title"], overview["overview"], report_sections)


class OpenAICompatibleVerificationProvider:
    name = "OpenAI-compatible"

    def __init__(self, model: str, base_url: str, api_key: str | None = None):
        self.model, self.client = model, _ChatClient(model, base_url, api_key)

    def verify(self, aligned_sections_path: str, draft_report_path: str, checkpoint_path: str):
        from src.synthesis.synthesizer import _validate_numeric_grounding
        source = json.loads(Path(aligned_sections_path).read_text(encoding="utf-8"))
        draft = json.loads(Path(draft_report_path).read_text(encoding="utf-8"))
        verified, audit = [], []
        for evidence, section in zip(source, draft["sections"], strict=True):
            data = self.client.generate_json("Verify the draft only against the evidence. Return only JSON with title, summary, key_points, is_faithful, issues.\nEvidence:\n" + json.dumps(evidence, ensure_ascii=False) + "\nDraft:\n" + json.dumps(section, ensure_ascii=False))
            trusted = evidence["text"] + "\n" + "\n".join(v.get("visible_text", "") for v in evidence["visuals"])
            selected = data
            try:
                _validate_numeric_grounding("\n".join([data["title"], data["summary"], *data["key_points"]]), trusted, "Compatible verification")
            except ValueError as error:
                selected, data["is_faithful"], data["issues"] = section, False, [f"Verifier rewrite rejected: {error}"]
            verified.append(ReportSection(title=selected["title"], start=float(evidence["start"]), end=float(evidence["end"]), summary=selected["summary"], key_points=selected["key_points"], visual_timestamps=[float(v["timestamp"]) for v in evidence["visuals"]]))
            audit.append({"section_index": evidence["section_index"], "is_faithful": data["is_faithful"], "issues": data["issues"]})
        return KnowledgeReport(draft["title"], draft["overview"], verified), audit
