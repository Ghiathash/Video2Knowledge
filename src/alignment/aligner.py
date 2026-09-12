from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.schemas.alignment import (
    AlignedSection,
    AlignedVisual,
)
from src.schemas.section import TranscriptSection
from src.schemas.visual import VisualAnalysis


def _visual_text(
    visual: VisualAnalysis,
) -> str:

    parts = [
        visual.description,
        visual.visible_text,
        " ".join(visual.knowledge_points),
        visual.transcript_alignment,
    ]

    return " ".join(
        part
        for part in parts
        if part
    )


def _semantic_similarity(
    visual: VisualAnalysis,
    section: TranscriptSection,
) -> float:

    visual_text = _visual_text(
        visual
    ).strip()

    section_text = section.text.strip()

    if not visual_text or not section_text:
        return 0.0

    try:
        matrix = TfidfVectorizer(
            stop_words="english"
        ).fit_transform(
            [
                visual_text,
                section_text,
            ]
        )

    except ValueError:
        return 0.0

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:2],
    )[0][0]

    return float(similarity)


def _find_containing_section(
    sections: list[TranscriptSection],
    timestamp: float,
) -> int | None:

    for index, section in enumerate(
        sections
    ):
        if (
            section.start
            <= timestamp
            <= section.end
        ):
            return index

    return None


def _candidate_sections(
    sections: list[TranscriptSection],
    timestamp: float,
    boundary_window: float,
) -> list[int]:

    containing = _find_containing_section(
        sections,
        timestamp,
    )

    if containing is None:

        nearest = min(
            range(len(sections)),
            key=lambda index: min(
                abs(
                    timestamp
                    - sections[index].start
                ),
                abs(
                    timestamp
                    - sections[index].end
                ),
            ),
        )

        return [nearest]

    candidates = {
        containing
    }

    current = sections[containing]

    # Visual can change shortly BEFORE
    # the spoken topic changes.
    if (
        containing < len(sections) - 1
        and current.end - timestamp
        <= boundary_window
    ):
        candidates.add(
            containing + 1
        )

    # Or shortly AFTER the spoken topic changed.
    if (
        containing > 0
        and timestamp - current.start
        <= boundary_window
    ):
        candidates.add(
            containing - 1
        )

    return sorted(candidates)


def _temporal_score(
    section: TranscriptSection,
    timestamp: float,
    boundary_window: float,
) -> float:

    if (
        section.start
        <= timestamp
        <= section.end
    ):
        return 1.0

    distance = min(
        abs(timestamp - section.start),
        abs(timestamp - section.end),
    )

    return max(
        0.0,
        1.0 - (
            distance
            / boundary_window
        ),
    )


def align_visuals_to_sections(
    sections: list[TranscriptSection],
    visuals: list[VisualAnalysis],
    boundary_window: float = 5.0,
    semantic_weight: float = 0.8,
) -> list[AlignedSection]:

    aligned_sections = [
        AlignedSection(
            section_index=index + 1,
            start=section.start,
            end=section.end,
            text=section.text,
            visuals=[],
        )
        for index, section
        in enumerate(sections)
    ]

    temporal_weight = (
        1.0 - semantic_weight
    )

    for visual in visuals:

        if not visual.contains_knowledge:
            continue

        candidates = _candidate_sections(
            sections,
            visual.timestamp,
            boundary_window,
        )

        # No ambiguity:
        # timestamp clearly belongs to one section.
        if len(candidates) == 1:

            selected_index = candidates[0]

            aligned_sections[
                selected_index
            ].visuals.append(
                AlignedVisual(
                    visual=visual,
                    alignment_score=1.0,
                    alignment_method="temporal",
                )
            )

            continue

        best_index = None
        best_score = -1.0

        for section_index in candidates:

            section = sections[
                section_index
            ]

            semantic_score = (
                _semantic_similarity(
                    visual,
                    section,
                )
            )

            temporal_score = (
                _temporal_score(
                    section,
                    visual.timestamp,
                    boundary_window,
                )
            )

            combined_score = (
                semantic_weight
                * semantic_score
                + temporal_weight
                * temporal_score
            )

            if combined_score > best_score:

                best_score = combined_score
                best_index = section_index

        aligned_sections[
            best_index
        ].visuals.append(
            AlignedVisual(
                visual=visual,
                alignment_score=float(
                    best_score
                ),
                alignment_method=(
                    "hybrid_temporal_semantic"
                ),
            )
        )

    return aligned_sections