from pathlib import Path

import cv2
import numpy as np

from src.schemas.visual import CandidateFrame


def _prepare_frame(frame):
    frame = cv2.resize(
        frame,
        (320, 180),
    )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    return gray


def _calculate_change_score(
    previous_frame,
    current_frame,
) -> float:
    difference = cv2.absdiff(
        previous_frame,
        current_frame,
    )

    score = np.mean(difference) / 255.0

    return float(score)


def detect_candidate_frames(
    video_path: str | Path,
    output_dir: str | Path,
    sample_interval: float = 1.0,
    change_threshold: float = 0.10,
    min_candidate_gap: float = 2.0,
) -> list[CandidateFrame]:

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        capture.release()
        raise ValueError(
            "Could not determine video FPS."
        )

    sample_every_frames = max(
        1,
        int(round(fps * sample_interval)),
    )

    candidates = []

    previous_processed = None
    last_candidate_timestamp = float("-inf")

    frame_index = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        if frame_index % sample_every_frames != 0:
            frame_index += 1
            continue

        timestamp = frame_index / fps

        processed = _prepare_frame(frame)

        if previous_processed is None:
            change_score = 1.0
        else:
            change_score = _calculate_change_score(
                previous_processed,
                processed,
            )

        enough_gap = (
            timestamp - last_candidate_timestamp
            >= min_candidate_gap
        )

        significant_change = (
            change_score >= change_threshold
        )

        if significant_change and enough_gap:
            output_path = (
                output_dir
                / f"candidate_{timestamp:.2f}.jpg"
            )

            cv2.imwrite(
                str(output_path),
                frame,
            )

            candidates.append(
    CandidateFrame(
        path=output_path,
        timestamp=timestamp,
        change_score=change_score,
        sources=["scene_change"],
    )
)

            last_candidate_timestamp = timestamp

        previous_processed = processed

        frame_index += 1

    capture.release()

    return candidates