from src.schemas.visual import CandidateFrame


def merge_candidates(
    scene_candidates: list[CandidateFrame],
    transcript_candidates: list[CandidateFrame],
    video_duration: float,
    tolerance: float = 2.0,
    start_margin: float = 1.0,
    end_margin: float = 2.0,
) -> list[CandidateFrame]:

    all_candidates = (
        scene_candidates
        + transcript_candidates
    )

    all_candidates.sort(
        key=lambda item: item.timestamp
    )

    merged: list[CandidateFrame] = []

    for candidate in all_candidates:

        if candidate.timestamp <= start_margin:
            continue

        if candidate.timestamp >= (
            video_duration - end_margin
        ):
            continue

        duplicate = None

        for existing in merged:
            if abs(
                existing.timestamp
                - candidate.timestamp
            ) <= tolerance:
                duplicate = existing
                break

        if duplicate is None:
            merged.append(candidate)
            continue

        existing_has_scene = (
            "scene_change"
            in duplicate.sources
        )

        candidate_has_scene = (
            "scene_change"
            in candidate.sources
        )

        combined_sources = list(
            dict.fromkeys(
                duplicate.sources
                + candidate.sources
            )
        )

        if (
            candidate_has_scene
            and not existing_has_scene
        ):
            duplicate.path = candidate.path
            duplicate.timestamp = candidate.timestamp
            duplicate.change_score = candidate.change_score

        duplicate.sources = combined_sources

        if (
            duplicate.evidence is None
            and candidate.evidence is not None
        ):
            duplicate.evidence = candidate.evidence

    return merged