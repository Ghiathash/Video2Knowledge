from pathlib import Path

from src.ingestion.validator import validate_video_path
from src.ingestion.metadata import extract_video_metadata
from src.schemas.video import VideoInfo


def load_video(video_path: str | Path) -> VideoInfo:
    validated_path = validate_video_path(video_path)

    video_info = extract_video_metadata(validated_path)

    return video_info