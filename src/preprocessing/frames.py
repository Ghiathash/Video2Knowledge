import subprocess
from pathlib import Path

from src.schemas.frame import FrameInfo


def extract_frame(
    video_path: str | Path,
    timestamp: float,
    output_path: str | Path,
) -> Path:
    video_path = Path(video_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(timestamp),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(output_path),
    ]

    subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    return output_path


def sample_frames(
    video_path: str | Path,
    duration: float,
    interval: float,
    output_dir: str | Path,
) -> list[FrameInfo]:

    if interval <= 0:
        raise ValueError("Sampling interval must be greater than 0.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []

    timestamp = 0.0

    # Container durations can extend a few milliseconds beyond the last
    # decodable video frame. Keep a small safety margin at the stream end.
    while timestamp < max(0.0, duration - 0.05):
        output_path = output_dir / f"frame_{timestamp:.2f}.jpg"

        extract_frame(
            video_path=video_path,
            timestamp=timestamp,
            output_path=output_path,
        )

        frames.append(
            FrameInfo(
                path=output_path,
                timestamp=timestamp,
            )
        )

        timestamp += interval

    return frames
