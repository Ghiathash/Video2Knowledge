from src.schemas.transcript import TranscriptSegment


def get_transcript_context(
    segments: list[TranscriptSegment],
    timestamp: float,
    window: float = 10.0,
) -> str:

    start_time = timestamp - window
    end_time = timestamp + window

    relevant_segments = [
        segment
        for segment in segments
        if segment.end >= start_time
        and segment.start <= end_time
    ]

    return " ".join(
        segment.text
        for segment in relevant_segments
    )