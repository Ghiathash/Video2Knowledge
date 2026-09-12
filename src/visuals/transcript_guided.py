import re
from pathlib import Path

import cv2

from src.schemas.transcript import TranscriptSegment
from src.schemas.visual import CandidateFrame


VISUAL_CUE_PATTERNS = {
    "table": r"\btables?\b",
    "figure": r"\bfigures?\b",
    "diagram": r"\bdiagrams?\b",
    "chart": r"\bcharts?\b",
    "graph": r"\bgraphs?\b",
    "slide": r"\bslides?\b",
    "equation": r"\bequations?\b",
    "formula": r"\bformulas?\b",
    "flowchart": r"\bflowcharts?\b",
    "architecture": r"\barchitecture\b",
    "whiteboard": r"\bwhiteboards?\b",
    "screen": r"\bscreens?\b",
}


def _find_visual_cues(
    text: str,
) -> list[str]:

    normalized = text.lower()

    found_cues = []

    for cue_name, pattern in VISUAL_CUE_PATTERNS.items():

        if re.search(
            pattern,
            normalized,
        ):
            found_cues.append(
                cue_name
            )

    return found_cues


def extract_transcript_guided_candidates(
    video_path: str | Path,
    transcript_segments: list[TranscriptSegment],
    output_dir: str | Path,
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

    candidates = []

    for segment in transcript_segments:

        cues = _find_visual_cues(
            segment.text
        )

        if not cues:
            continue

        timestamp = (
            segment.start + segment.end
        ) / 2

        capture.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000,
        )

        success, frame = capture.read()

        if not success:
            continue

        output_path = (
            output_dir
            / f"transcript_{timestamp:.2f}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            frame,
        )

        candidates.append(
            CandidateFrame(
                path=output_path,
                timestamp=timestamp,
                change_score=0.0,
                sources=["transcript_cue"],
                evidence=segment.text,
            )
        )

    capture.release()

    return candidates