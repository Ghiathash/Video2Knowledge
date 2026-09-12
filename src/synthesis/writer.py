import json
from pathlib import Path

from src.schemas.report import KnowledgeReport


def save_knowledge_report(
    report: KnowledgeReport,
    output_path: str | Path,
) -> Path:

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "title": report.title,
        "overview": report.overview,
        "sections": [
            {
                "title": section.title,
                "start": section.start,
                "end": section.end,
                "summary": section.summary,
                "key_points": section.key_points,
                "visual_timestamps": section.visual_timestamps,
            }
            for section in report.sections
        ],
    }

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