import json
from pathlib import Path

from src.schemas.alignment import AlignedSection


def save_aligned_sections(
    sections: list[AlignedSection],
    output_path: str | Path,
) -> Path:

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = []

    for section in sections:

        visuals = []

        for aligned_visual in section.visuals:

            visual = aligned_visual.visual

            visuals.append(
                {
                    "path": str(visual.path),
                    "timestamp": visual.timestamp,
                    "visual_type": visual.visual_type,
                    "importance_score": visual.importance_score,
                    "description": visual.description,
                    "visible_text": visual.visible_text,
                    "knowledge_points": visual.knowledge_points,
                    "transcript_alignment": visual.transcript_alignment,
                    "alignment_score": aligned_visual.alignment_score,
                    "alignment_method": aligned_visual.alignment_method,
                }
            )

        data.append(
            {
                "section_index": section.section_index,
                "start": section.start,
                "end": section.end,
                "text": section.text,
                "visuals": visuals,
            }
        )

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