import cv2
import numpy as np

from src.schemas.visual import CandidateFrame


def _prepare_image(path):
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(
            f"Could not read image: {path}"
        )

    image = cv2.resize(
        image,
        (320, 180),
    )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


def _calculate_similarity(
    first_path,
    second_path,
) -> float:

    first = _prepare_image(first_path)
    second = _prepare_image(second_path)

    difference = cv2.absdiff(
        first,
        second,
    )

    similarity = (
        1.0
        - np.mean(difference) / 255.0
    )

    return float(similarity)


def deduplicate_candidates(
    candidates: list[CandidateFrame],
    similarity_threshold: float = 0.98,
) -> list[CandidateFrame]:

    unique: list[CandidateFrame] = []

    for candidate in candidates:

        duplicate = None

        for existing in unique:

            similarity = _calculate_similarity(
                existing.path,
                candidate.path,
            )

            if similarity >= similarity_threshold:
                duplicate = existing
                break

        if duplicate is None:
            unique.append(candidate)
            continue

        combined_sources = list(
            dict.fromkeys(
                duplicate.sources
                + candidate.sources
            )
        )

        best_change_score = max(
            duplicate.change_score,
            candidate.change_score,
        )

        candidate_has_transcript = (
            "transcript_cue"
            in candidate.sources
        )

        existing_has_transcript = (
            "transcript_cue"
            in duplicate.sources
        )

        if (
            candidate_has_transcript
            and not existing_has_transcript
        ):
            duplicate.path = candidate.path
            duplicate.timestamp = candidate.timestamp

        duplicate.change_score = best_change_score
        duplicate.sources = combined_sources

        if (
            candidate.evidence
            and not duplicate.evidence
        ):
            duplicate.evidence = candidate.evidence

    return unique