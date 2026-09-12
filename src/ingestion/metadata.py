import json
import subprocess
from pathlib import Path

from src.schemas.video import VideoInfo


def extract_video_metadata(video_path: Path) -> VideoInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    data = json.loads(result.stdout)

    streams = data.get("streams", [])
    format_info = data.get("format", {})

    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )

    audio_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        None,
    )

    if video_stream is None:
        raise ValueError(f"No video stream found in: {video_path}")

    duration = float(format_info.get("duration", 0))

    fps_raw = video_stream.get("r_frame_rate", "0/1")
    numerator, denominator = fps_raw.split("/")
    fps = float(numerator) / float(denominator) if float(denominator) != 0 else 0.0

    return VideoInfo(
        path=video_path,
        duration=duration,
        fps=fps,
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        video_codec=video_stream.get("codec_name", "unknown"),
        has_audio=audio_stream is not None,
        audio_codec=(
            audio_stream.get("codec_name")
            if audio_stream is not None
            else None
        ),
    )