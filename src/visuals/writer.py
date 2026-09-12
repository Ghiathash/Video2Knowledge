import json
from pathlib import Path

from src.schemas.visual import VisualAnalysis


def _analysis_to_dict(
    analysis: VisualAnalysis,
) -> dict:

    return {
        "path": str(analysis.path),
        "timestamp": analysis.timestamp,
        "visual_type": analysis.visual_type,
        "contains_knowledge": analysis.contains_knowledge,
        "importance_score": analysis.importance_score,
        "description": analysis.description,
        "visible_text": analysis.visible_text,
        "knowledge_points": analysis.knowledge_points,
        "transcript_alignment": analysis.transcript_alignment,
    }


def _dict_to_analysis(
    data: dict,
) -> VisualAnalysis:

    return VisualAnalysis(
        path=Path(data["path"]),
        timestamp=data["timestamp"],
        visual_type=data["visual_type"],
        contains_knowledge=data["contains_knowledge"],
        importance_score=data["importance_score"],
        description=data["description"],
        visible_text=data["visible_text"],
        knowledge_points=data["knowledge_points"],
        transcript_alignment=data["transcript_alignment"],
    )


def save_visual_analyses(
    analyses: list[VisualAnalysis],
    output_path: str | Path,
) -> Path:

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = [
        _analysis_to_dict(analysis)
        for analysis in analyses
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


def load_visual_analyses(
    input_path: str | Path,
) -> list[VisualAnalysis]:

    input_path = Path(input_path)

    if not input_path.exists():
        return []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return [
        _dict_to_analysis(item)
        for item in data
    ]


def upsert_visual_analysis(
    analysis: VisualAnalysis,
    output_path: str | Path,
    timestamp_tolerance: float = 0.5,
) -> Path:

    analyses = load_visual_analyses(
        output_path
    )

    replaced = False

    for index, existing in enumerate(
        analyses
    ):
        if abs(
            existing.timestamp
            - analysis.timestamp
        ) <= timestamp_tolerance:

            analyses[index] = analysis
            replaced = True
            break

    if not replaced:
        analyses.append(analysis)

    analyses.sort(
        key=lambda item: item.timestamp
    )

    return save_visual_analyses(
        analyses,
        output_path,
    )