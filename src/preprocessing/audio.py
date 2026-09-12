import subprocess
from pathlib import Path


def extract_audio(
    video_path: str | Path,
    output_path: str | Path,
) -> Path:
    video_path = Path(video_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]

    subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    return output_path