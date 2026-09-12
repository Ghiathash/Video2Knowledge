import mimetypes
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.schemas.visual import VisualAnalysis


load_dotenv()
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
)

class VisualAnalysisResponse(BaseModel):

    visual_type: Literal[
        "diagram",
        "flowchart",
        "architecture",
        "chart",
        "table",
        "equation",
        "code",
        "slide",
        "whiteboard",
        "ui",
        "person",
        "other",
    ]

    contains_knowledge: bool

    importance_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    description: str

    visible_text: str

    knowledge_points: list[str]

    transcript_alignment: str


def _get_client() -> genai.Client:

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found in the .env file."
        )

    return genai.Client(
        api_key=api_key
    )


def _get_mime_type(
    image_path: Path,
) -> str:

    mime_type, _ = mimetypes.guess_type(
        image_path
    )

    if mime_type is None:
        return "image/jpeg"

    return mime_type


def analyze_visual(
    image_path: str | Path,
    timestamp: float,
    transcript_context: str,
) -> VisualAnalysis:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image_bytes = image_path.read_bytes()

    mime_type = _get_mime_type(
        image_path
    )

    prompt = f"""
You are analyzing one candidate frame extracted from
an educational video.

The goal is to decide whether this frame contains
important VISUAL KNOWLEDGE that should appear in a
structured study report.

The frame timestamp is:
{timestamp:.2f} seconds

Transcript around this timestamp:
---
{transcript_context}
---

Analyze BOTH:
1. The actual visual content of the image.
2. Its relationship to the nearby transcript.

Possible primary visual types:
- diagram
- flowchart
- architecture
- chart
- table
- equation
- code
- slide
- whiteboard
- ui
- person
- other

Important rules:

- A talking person alone is usually NOT important
  visual knowledge.

- A title-only slide is usually low importance.

- Tables, diagrams, charts, equations, code,
  architecture drawings, explanatory slides,
  whiteboards, and meaningful UI screens may
  contain important knowledge.

- Do not mark a frame as important only because
  the transcript contains useful information.
  The IMAGE itself must contribute useful visual
  information.

- visible_text must contain only text you can
  actually read from the image.

- Do not invent text.

- knowledge_points should describe knowledge
  communicated by the visual.

- transcript_alignment should briefly explain
  how the visual supports or corresponds to the
  nearby transcript.

- importance_score must represent the importance
  of including THIS VISUAL in the final report.

Return concise results.
"""

    client = _get_client()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=VisualAnalysisResponse,
        ),
    )

    result = response.parsed

    if result is None:
        result = VisualAnalysisResponse.model_validate_json(
            response.text
        )

    return VisualAnalysis(
        path=image_path,
        timestamp=timestamp,
        visual_type=result.visual_type,
        contains_knowledge=result.contains_knowledge,
        importance_score=result.importance_score,
        description=result.description,
        visible_text=result.visible_text,
        knowledge_points=result.knowledge_points,
        transcript_alignment=result.transcript_alignment,
    )