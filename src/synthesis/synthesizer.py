import json
import hashlib
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from src.schemas.report import (
    KnowledgeReport,
    ReportSection,
)


load_dotenv()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
)


# -------------------------
# Structured responses
# -------------------------

class SectionSynthesisResponse(BaseModel):
    title: str
    summary: str
    key_points: list[str]


class ReportOverviewResponse(BaseModel):
    title: str
    overview: str


# -------------------------
# Numeric grounding
# -------------------------

NUMBER_PATTERN = re.compile(
    r"""
    (?<![\w])
    \$?
    \d[\d,]*
    (?:\.\d+)?
    (?:
        \s?
        (?:
            %
            |
            k
            |
            m
            |
            b
            |
            thousand
            |
            million
            |
            billion
        )
    )?
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _normalize_number(
    value: str,
) -> str:

    return (
        value
        .lower()
        .replace(",", "")
        .replace("$", "")
        .replace(" ", "")
    )


def _extract_numbers(
    text: str,
) -> set[str]:

    matches = NUMBER_PATTERN.findall(text)

    return {
        _normalize_number(match)
        for match in matches
    }


def _validate_numeric_grounding(
    generated_text: str,
    source_text: str,
    context_name: str,
) -> None:

    generated_numbers = _extract_numbers(
        generated_text
    )

    source_numbers = _extract_numbers(
        source_text
    )

    unsupported = (
        generated_numbers
        - source_numbers
    )

    if unsupported:
        raise ValueError(
            f"{context_name} contains unsupported "
            f"numeric values: {sorted(unsupported)}"
        )


def _has_only_grounded_numbers(text: str, source_numbers: set[str]) -> bool:
    return _extract_numbers(text).issubset(source_numbers)


def _sanitize_numeric_response(
    result: SectionSynthesisResponse,
    source_text: str,
) -> SectionSynthesisResponse:
    """Drop claims containing numbers that remain unsupported after retries."""
    source_numbers = _extract_numbers(source_text)
    title = result.title if _has_only_grounded_numbers(result.title, source_numbers) else "Key Concepts"
    sentences = re.split(r"(?<=[.!?])\s+", result.summary)
    summary = " ".join(
        sentence for sentence in sentences
        if _has_only_grounded_numbers(sentence, source_numbers)
    ).strip()
    if not summary:
        summary = "This section explains the key concepts presented in the lesson."
    key_points = [
        point for point in result.key_points
        if _has_only_grounded_numbers(point, source_numbers)
    ]
    if not key_points:
        key_points = [summary]
    return SectionSynthesisResponse(
        title=title,
        summary=summary,
        key_points=key_points,
    )


# -------------------------
# Helpers
# -------------------------

def _get_client() -> genai.Client:

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found."
        )

    return genai.Client(
        api_key=api_key
    )


def _build_section_evidence(
    section: dict[str, Any],
) -> dict:

    visuals = []

    for visual in section["visuals"]:

        visuals.append(
            {
                "visual_type": (
                    visual["visual_type"]
                ),
                "description": (
                    visual["description"]
                ),
                "visible_text": (
                    visual["visible_text"]
                ),
                "knowledge_points": (
                    visual["knowledge_points"]
                ),
                "transcript_alignment": (
                    visual[
                        "transcript_alignment"
                    ]
                ),
            }
        )

    return {
        "section_index": (
            section["section_index"]
        ),
        "transcript": (
            section["text"]
        ),
        "visual_evidence": visuals,
    }


def _section_source_text(
    evidence: dict,
) -> str:

    parts = [
        evidence["transcript"]
    ]

    for visual in evidence[
        "visual_evidence"
    ]:

        parts.extend(
            [
                visual["description"],
                visual["visible_text"],
                " ".join(
                    visual[
                        "knowledge_points"
                    ]
                ),
                visual[
                    "transcript_alignment"
                ],
            ]
        )

    return "\n".join(
        part
        for part in parts
        if part
    )


# -------------------------
# Section synthesis
# -------------------------

def _synthesize_section(
    client: genai.Client,
    section: dict[str, Any],
) -> ReportSection:

    section_index = section[
        "section_index"
    ]

    evidence = _build_section_evidence(
        section
    )

    source_text = _section_source_text(
        evidence
    )

    prompt = f"""
You are creating ONE section of a factual
study report from an educational video.

You have access ONLY to the evidence for
this section.

SECTION EVIDENCE:
{json.dumps(
    evidence,
    ensure_ascii=False,
    indent=2
)}

STRICT RULES:

1. Use ONLY the evidence shown above.

2. You have no knowledge of the other sections.
   Do not infer what may appear before or after
   this section.

3. Do not invent facts.

4. Do not infer facts that are not explicitly
   supported.

5. Do not strengthen claims.

Example:
"occurrence of a heart attack"
must not become
"first occurrence of a heart attack"
unless the word "first" is supported.

6. Numerical values are critical.

Every number, percentage, threshold, range,
count, duration, or quantity in your output
must already appear in this section's evidence.

Do NOT calculate new numbers.

Do NOT estimate.

Do NOT change precise numbers into different
numbers.

7. If transcript evidence and visual evidence
   complement each other, combine them carefully.

8. Visual evidence may provide useful knowledge
   that is not fully spoken in the transcript,
   but use it only when explicitly present in
   the supplied visual evidence.

9. Write a clear descriptive title.

10. Write a concise but informative summary.

11. Extract the most important study points.

12. Do not mention timestamps.

13. Do not mention that you are using a
    transcript, image, model, or source data.

Return only:
- title
- summary
- key_points
"""

    for attempt in range(3):
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=SectionSynthesisResponse,
            ),
        )
        result = response.parsed
        if result is None:
            result = SectionSynthesisResponse.model_validate_json(response.text)

        generated_text = "\n".join(
            [result.title, result.summary, *result.key_points]
        )
        try:
            _validate_numeric_grounding(
                generated_text=generated_text,
                source_text=source_text,
                context_name=f"Section {section_index}",
            )
            break
        except ValueError as error:
            if attempt == 2:
                result = _sanitize_numeric_response(result, source_text)
                generated_text = "\n".join(
                    [result.title, result.summary, *result.key_points]
                )
                _validate_numeric_grounding(
                    generated_text, source_text, f"Section {section_index}"
                )
                break
            prompt += (
                "\n\nYour previous response was rejected: "
                f"{error}. Regenerate without unsupported numeric values."
            )

    # Timestamps are NOT generated by Gemini.
    # They come directly from our aligned data.
    visual_timestamps = [
        float(visual["timestamp"])
        for visual in section["visuals"]
    ]

    return ReportSection(
        title=result.title,
        start=float(section["start"]),
        end=float(section["end"]),
        summary=result.summary,
        key_points=result.key_points,
        visual_timestamps=(
            visual_timestamps
        ),
    )


# -------------------------
# Final title + overview
# -------------------------

def _generate_overview(
    client: genai.Client,
    sections: list[ReportSection],
) -> ReportOverviewResponse:

    grounded_sections = []

    for index, section in enumerate(
        sections,
        start=1,
    ):

        grounded_sections.append(
            {
                "section_index": index,
                "title": section.title,
                "summary": section.summary,
                "key_points": (
                    section.key_points
                ),
            }
        )

    source_text = json.dumps(
        grounded_sections,
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
Create the overall title and overview for
a study report.

The content below has already been generated
and validated section by section.

VALIDATED SECTIONS:
{source_text}

STRICT RULES:

1. Use ONLY these validated sections.

2. Do not introduce new facts.

3. Do not introduce new numerical values.

4. Do not infer additional results,
   conclusions, or methodology.

5. The title should accurately represent
   the main subject.

6. The overview should briefly explain
   what the report covers.

7. Avoid unnecessary detail.

Return only:
- title
- overview
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type=(
                "application/json"
            ),
            response_schema=(
                ReportOverviewResponse
            ),
        ),
    )

    result = response.parsed

    if result is None:
        result = (
            ReportOverviewResponse
            .model_validate_json(
                response.text
            )
        )

    generated_text = (
        result.title
        + "\n"
        + result.overview
    )

    _validate_numeric_grounding(
        generated_text=generated_text,
        source_text=source_text,
        context_name="Report overview",
    )

    return result


# -------------------------
# Public function
# -------------------------

def synthesize_report(
    aligned_sections_path: str,
    checkpoint_path: str | None = None,
) -> KnowledgeReport:

    with open(
        aligned_sections_path,
        "r",
        encoding="utf-8",
    ) as file:

        aligned_sections = json.load(
            file
        )

    if not aligned_sections:
        raise ValueError(
            "No aligned sections found."
        )

    client = _get_client()

    fingerprint = hashlib.sha256(
        json.dumps(aligned_sections, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    checkpoint_file = Path(checkpoint_path) if checkpoint_path else None
    report_sections = []
    if checkpoint_file and checkpoint_file.exists():
        try:
            saved = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            if saved.get("fingerprint") == fingerprint:
                report_sections = [ReportSection(**item) for item in saved.get("sections", [])]
        except (OSError, json.JSONDecodeError, TypeError):
            report_sections = []

    total = len(
        aligned_sections
    )

    for index, section in enumerate(
        aligned_sections,
        start=1,
    ):

        if index <= len(report_sections):
            continue

        print(
            f"Synthesizing section "
            f"{index}/{total}..."
        )

        report_section = (
            _synthesize_section(
                client=client,
                section=section,
            )
        )

        report_sections.append(
            report_section
        )
        if checkpoint_file:
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_file.write_text(
                json.dumps(
                    {
                        "fingerprint": fingerprint,
                        "sections": [vars(item) for item in report_sections],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    print(
        "Generating report overview..."
    )

    overview = _generate_overview(
        client=client,
        sections=report_sections,
    )

    report = KnowledgeReport(
        title=overview.title,
        overview=overview.overview,
        sections=report_sections,
    )
    if checkpoint_file and checkpoint_file.exists():
        checkpoint_file.unlink()
    return report
