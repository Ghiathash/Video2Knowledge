import hashlib
import json
import os
import re
from decimal import Decimal
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


# =========================================================
# Response schemas
# =========================================================

class SectionVerificationResponse(BaseModel):
    title: str
    summary: str
    key_points: list[str]
    is_faithful: bool
    issues: list[str]


class OverviewVerificationResponse(BaseModel):
    title: str
    overview: str
    issues: list[str]


# =========================================================
# Numeric normalization
# =========================================================

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


NUMBER_PATTERN = re.compile(
    r"""
    (?<!\w)
    \$?
    (?P<number>\d[\d,]*(?:\.\d+)?)
    (?:
        \s?
        (?P<suffix>
            %
            |
            k\b
            |
            m\b
            |
            b\b
            |
            thousand\b
            |
            million\b
            |
            billion\b
        )
    )?
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _normalize_number_words(
    text: str,
) -> str:

    normalized = text.lower()

    for word, number in NUMBER_WORDS.items():
        normalized = re.sub(
            rf"\b{word}\b",
            number,
            normalized,
        )

    return normalized


def _canonical_number(
    number: str,
    suffix: str | None,
) -> str:

    value = Decimal(
        number.replace(",", "")
    )

    if suffix:

        suffix = suffix.lower()

        if suffix in {
            "k",
            "thousand",
        }:
            value *= Decimal("1000")

        elif suffix in {
            "m",
            "million",
        }:
            value *= Decimal(
                "1000000"
            )

        elif suffix in {
            "b",
            "billion",
        }:
            value *= Decimal(
                "1000000000"
            )

        # % is intentionally treated as the
        # same numeric value for grounding.
        #
        # 67.56
        # 67.56%
        #
        # both become 67.56.

    result = format(
        value,
        "f",
    )

    if "." in result:
        result = (
            result.rstrip("0")
            .rstrip(".")
        )

    return result


def _extract_numbers(
    text: str,
) -> set[str]:

    normalized_text = (
        _normalize_number_words(
            text
        )
    )

    numbers = set()

    for match in (
        NUMBER_PATTERN.finditer(
            normalized_text
        )
    ):

        number = (
            _canonical_number(
                match.group(
                    "number"
                ),
                match.group(
                    "suffix"
                ),
            )
        )

        numbers.add(number)

    return numbers


def _validate_numbers(
    generated_text: str,
    trusted_source: str,
    context_name: str,
) -> None:

    generated_numbers = (
        _extract_numbers(
            generated_text
        )
    )

    source_numbers = (
        _extract_numbers(
            trusted_source
        )
    )

    unsupported = (
        generated_numbers
        - source_numbers
    )

    if unsupported:
        raise ValueError(
            f"{context_name} contains "
            f"unsupported numbers: "
            f"{sorted(unsupported)}"
        )


# =========================================================
# Gemini
# =========================================================

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


# =========================================================
# Evidence
# =========================================================

def _build_evidence(
    section: dict[str, Any],
) -> dict:

    visuals = []

    for visual in section[
        "visuals"
    ]:

        visuals.append(
            {
                "timestamp": (
                    visual["timestamp"]
                ),
                "visual_type": (
                    visual[
                        "visual_type"
                    ]
                ),

                # Strong evidence
                "visible_text": (
                    visual[
                        "visible_text"
                    ]
                ),

                # Lower-trust interpretation
                "description": (
                    visual[
                        "description"
                    ]
                ),
                "knowledge_points": (
                    visual[
                        "knowledge_points"
                    ]
                ),
            }
        )

    return {
        "transcript": (
            section["text"]
        ),
        "visuals": visuals,
    }


def _trusted_source_text(
    evidence: dict,
) -> str:

    # Numeric grounding only trusts:
    #
    # 1. Transcript
    # 2. Text explicitly visible in image

    parts = [
        evidence["transcript"]
    ]

    for visual in evidence[
        "visuals"
    ]:

        parts.append(
            visual[
                "visible_text"
            ]
        )

    return "\n".join(
        part
        for part in parts
        if part
    )


# =========================================================
# Section verification
# =========================================================

def _verify_section(
    client: genai.Client,
    source_section: dict,
    draft_section: dict,
) -> tuple[ReportSection, dict]:

    section_index = (
        source_section[
            "section_index"
        ]
    )

    evidence = _build_evidence(
        source_section
    )

    prompt = f"""
You are a strict faithfulness verifier.

Your job is NOT to expand the report.

Your job is to verify and correct the draft so
that every factual claim is supported by the
supplied evidence.

EVIDENCE:
{json.dumps(
    evidence,
    ensure_ascii=False,
    indent=2
)}

DRAFT REPORT SECTION:
{json.dumps(
    draft_section,
    ensure_ascii=False,
    indent=2
)}

EVIDENCE PRIORITY:

1. Explicit transcript statements have the
   highest priority when determining semantic
   relationships such as:

   - AND
   - OR
   - sequence
   - conditions
   - requirements
   - definitions
   - comparisons
   - causality
   - timing

2. Text explicitly visible in the image is also
   strong factual evidence.

3. VLM descriptions and knowledge points are
   lower-trust interpretations.

IMPORTANT EVIDENCE RULE:

Absence of a fact from the transcript is NOT
a contradiction.

If the transcript does not mention a fact,
but the visible image text explicitly states it,
that fact may be included.

Example:

Transcript:
"The target is the occurrence of a heart attack."

Visible text:
"The target prediction is the first occurrence
of a heart attack."

The word "first" is allowed because it is
explicitly supported by visible text and does
not contradict the transcript.

However, if transcript and visual evidence
explicitly contradict each other about a
semantic relationship, prefer the explicit
transcript statement.

For example:

Transcript:
"A alongside B followed by C"

must NOT become:

"A OR B"

simply because the slide visually lists the
items separately.

STRICT RULES:

- Remove unsupported claims.

- Correct changed semantic relationships.

- Do not change AND into OR.

- Do not change OR into AND.

- Facts explicitly stated in visible text are
  allowed even when they are not spoken.

- Do not treat transcript silence as evidence
  against visible text.

- Do not introduce words such as:
  only
  always
  never
  causes
  proves
  first
  highest
  lowest

  unless they are explicitly supported by
  either the transcript OR visible text.

- Preserve supported numerical values exactly.

- Do not calculate new numerical values.

- Do not estimate.

- Do not import knowledge from other sections.

- VLM interpretations must never override
  explicit transcript or visible text.

- If the draft is already faithful, preserve
  its meaning.

Return:
- corrected title
- corrected summary
- corrected key_points
- is_faithful
- issues

issues must briefly list actual faithfulness
problems found.

Do NOT report a fact as unsupported merely
because it appears only in visible image text.

If no problem is found, return an empty issues list.
"""
    response = (
        client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=(
                types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type=(
                        "application/json"
                    ),
                    response_schema=(
                        SectionVerificationResponse
                    ),
                )
            ),
        )
    )

    result = response.parsed

    if result is None:
        result = (
            SectionVerificationResponse
            .model_validate_json(
                response.text
            )
        )

    generated_text = "\n".join(
        [
            result.title,
            result.summary,
            *result.key_points,
        ]
    )

    try:
        _validate_numbers(
            generated_text=generated_text,
            trusted_source=_trusted_source_text(evidence),
            context_name=f"Section {section_index}",
        )
    except ValueError as error:
        # The draft already passed synthesis numeric grounding. Never accept a
        # verifier rewrite that introduces new quantities.
        result = SectionVerificationResponse(
            title=draft_section["title"],
            summary=draft_section["summary"],
            key_points=draft_section["key_points"],
            is_faithful=False,
            issues=[f"Verifier rewrite rejected: {error}"],
        )

    visual_timestamps = [
        float(
            visual["timestamp"]
        )
        for visual
        in source_section[
            "visuals"
        ]
    ]

    verified_section = (
        ReportSection(
            title=result.title,
            start=float(
                source_section[
                    "start"
                ]
            ),
            end=float(
                source_section[
                    "end"
                ]
            ),
            summary=result.summary,
            key_points=(
                result.key_points
            ),
            visual_timestamps=(
                visual_timestamps
            ),
        )
    )

    audit = {
        "section_index": (
            section_index
        ),
        "is_faithful": (
            result.is_faithful
        ),
        "issues": (
            result.issues
        ),
    }

    return (
        verified_section,
        audit,
    )


# =========================================================
# Overview verification
# =========================================================

def _verify_overview(
    client: genai.Client,
    title: str,
    overview: str,
    verified_sections: list[
        ReportSection
    ],
) -> OverviewVerificationResponse:

    source = []

    for section in (
        verified_sections
    ):

        source.append(
            {
                "title": (
                    section.title
                ),
                "summary": (
                    section.summary
                ),
                "key_points": (
                    section.key_points
                ),
            }
        )

    prompt = f"""
Verify the following report title and overview
against the already verified report sections.

VERIFIED SECTIONS:
{json.dumps(
    source,
    ensure_ascii=False,
    indent=2
)}

CURRENT TITLE:
{title}

CURRENT OVERVIEW:
{overview}

STRICT RULES:

- Use only information present in the
  verified sections.

- Remove unsupported claims.

- Do not add new conclusions.

- Do not introduce new numerical values.

- Do not strengthen claims.

- Keep the title accurate.

- Keep the overview concise.

Return:
- corrected title
- corrected overview
- issues

If no issues exist, issues must be empty.
"""

    response = (
        client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=(
                types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type=(
                        "application/json"
                    ),
                    response_schema=(
                        OverviewVerificationResponse
                    ),
                )
            ),
        )
    )

    result = response.parsed

    if result is None:
        result = (
            OverviewVerificationResponse
            .model_validate_json(
                response.text
            )
        )

    source_text = json.dumps(
        source,
        ensure_ascii=False,
    )

    generated_text = (
        result.title
        + "\n"
        + result.overview
    )

    try:
        _validate_numbers(
            generated_text=generated_text,
            trusted_source=source_text,
            context_name="Report overview",
        )
    except ValueError as error:
        result = OverviewVerificationResponse(
            title=title,
            overview=overview,
            issues=[f"Verifier rewrite rejected: {error}"],
        )

    return result


# =========================================================
# Checkpoint helpers
# =========================================================

def _section_to_dict(
    section: ReportSection,
) -> dict:

    return {
        "title": section.title,
        "start": section.start,
        "end": section.end,
        "summary": section.summary,
        "key_points": (
            section.key_points
        ),
        "visual_timestamps": (
            section.visual_timestamps
        ),
    }


def _dict_to_section(
    data: dict,
) -> ReportSection:

    return ReportSection(
        title=data["title"],
        start=float(
            data["start"]
        ),
        end=float(
            data["end"]
        ),
        summary=data["summary"],
        key_points=(
            data["key_points"]
        ),
        visual_timestamps=[
            float(x)
            for x in data[
                "visual_timestamps"
            ]
        ],
    )


def _make_fingerprint(
    source_sections: list[dict],
    draft_sections: list[dict],
) -> str:

    payload = json.dumps(
        {
            "source": (
                source_sections
            ),
            "draft": (
                draft_sections
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def _save_checkpoint(
    checkpoint_path: Path,
    fingerprint: str,
    verified_sections: list[
        ReportSection
    ],
    audit: list[dict],
) -> None:

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "fingerprint": fingerprint,
        "verified_sections": [
            _section_to_dict(section)
            for section
            in verified_sections
        ],
        "audit": audit,
    }

    with checkpoint_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def _load_checkpoint(
    checkpoint_path: Path,
    fingerprint: str,
) -> tuple[
    list[ReportSection],
    list[dict],
]:

    if not checkpoint_path.exists():
        return [], []

    try:

        with checkpoint_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return [], []

    if (
        data.get("fingerprint")
        != fingerprint
    ):
        return [], []

    sections = [
        _dict_to_section(item)
        for item
        in data.get(
            "verified_sections",
            [],
        )
    ]

    audit = data.get(
        "audit",
        [],
    )

    return sections, audit


# =========================================================
# Public verifier
# =========================================================

def verify_report(
    aligned_sections_path: str,
    draft_report_path: str,
    checkpoint_path: str | None = None,
) -> tuple[
    KnowledgeReport,
    list[dict],
]:

    with open(
        aligned_sections_path,
        "r",
        encoding="utf-8",
    ) as file:

        source_sections = (
            json.load(file)
        )

    with open(
        draft_report_path,
        "r",
        encoding="utf-8",
    ) as file:

        draft_report = (
            json.load(file)
        )

    draft_sections = (
        draft_report[
            "sections"
        ]
    )

    if len(source_sections) != len(
        draft_sections
    ):
        raise ValueError(
            "Source and draft section "
            "counts do not match."
        )

    if checkpoint_path is None:
        raise ValueError(
            "checkpoint_path must be inside the current run directory."
        )

    checkpoint_file = Path(checkpoint_path)

    fingerprint = (
        _make_fingerprint(
            source_sections,
            draft_sections,
        )
    )

    verified_sections, audit = (
        _load_checkpoint(
            checkpoint_file,
            fingerprint,
        )
    )

    completed = len(
        verified_sections
    )

    if completed > 0:
        print(
            f"Resuming from checkpoint: "
            f"{completed}/"
            f"{len(source_sections)} "
            f"sections already verified."
        )

    client = _get_client()

    total = len(
        source_sections
    )

    for index in range(
        completed,
        total,
    ):

        source = (
            source_sections[
                index
            ]
        )

        draft = (
            draft_sections[
                index
            ]
        )

        print(
            f"Verifying section "
            f"{index + 1}/{total}..."
        )

        verified, record = (
            _verify_section(
                client=client,
                source_section=source,
                draft_section=draft,
            )
        )

        verified_sections.append(
            verified
        )

        audit.append(
            record
        )

        # Save immediately after every
        # successfully verified section.
        _save_checkpoint(
            checkpoint_path=(
                checkpoint_file
            ),
            fingerprint=(
                fingerprint
            ),
            verified_sections=(
                verified_sections
            ),
            audit=audit,
        )

    print(
        "Verifying report overview..."
    )

    overview = _verify_overview(
        client=client,
        title=(
            draft_report[
                "title"
            ]
        ),
        overview=(
            draft_report[
                "overview"
            ]
        ),
        verified_sections=(
            verified_sections
        ),
    )

    final_audit = (
        audit
        + [
            {
                "report_overview": True,
                "issues": (
                    overview.issues
                ),
            }
        ]
    )

    report = KnowledgeReport(
        title=overview.title,
        overview=overview.overview,
        sections=verified_sections,
    )

    # Verification fully succeeded.
    # Checkpoint is no longer required.
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    return report, final_audit
