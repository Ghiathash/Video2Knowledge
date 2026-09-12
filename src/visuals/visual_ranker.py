from pathlib import Path

import cv2
import numpy as np

from src.schemas.visual import CandidateFrame, RankedVisual


def _calculate_visual_richness(
    image_path: str | Path,
) -> float:
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.resize(
        gray,
        (320, 180),
    )

    mean_brightness = float(
        np.mean(gray)
    )

    contrast = float(
        np.std(gray)
    )

    # Detect almost blank black/white frames.
    if contrast < 5:
        return 0.0

    if mean_brightness < 5:
        return 0.0

    if mean_brightness > 250:
        return 0.0

    edges = cv2.Canny(
        gray,
        80,
        160,
    )

    edge_density = float(
        np.count_nonzero(edges)
        / edges.size
    )

    contrast_score = min(
        contrast / 64.0,
        1.0,
    )

    edge_score = min(
        edge_density / 0.12,
        1.0,
    )

    richness_score = (
        0.5 * contrast_score
        + 0.5 * edge_score
    )

    return float(richness_score)


def _normalize_change_score(
    change_score: float,
) -> float:
    return min(
        change_score / 0.10,
        1.0,
    )


def rank_candidate_frames(
    candidates: list[CandidateFrame],
    video_duration: float,
    end_margin: float = 2.0,
    top_k: int | None = None,
) -> list[RankedVisual]:

    ranked_visuals = []

    for candidate in candidates:

        # Ignore the artificial first-frame candidate.
        if candidate.timestamp <= 0.5:
            continue

        # Ignore frames extremely close to video ending.
        if candidate.timestamp >= (
            video_duration - end_margin
        ):
            continue

        visual_richness = (
            _calculate_visual_richness(
                candidate.path
            )
        )

        # Blank or almost empty visual.
        if visual_richness == 0:
            continue

        normalized_change = (
            _normalize_change_score(
                candidate.change_score
            )
        )

        importance_score = (
            0.55 * visual_richness
            + 0.45 * normalized_change
        )

        ranked_visuals.append(
            RankedVisual(
                path=candidate.path,
                timestamp=candidate.timestamp,
                change_score=candidate.change_score,
                visual_richness_score=visual_richness,
                importance_score=float(
                    importance_score
                ),
            )
        )

    ranked_visuals.sort(
        key=lambda item: item.importance_score,
        reverse=True,
    )

    if top_k is not None:
        ranked_visuals = ranked_visuals[:top_k]

    return ranked_visuals