from pathlib import Path

import pytest

from src.ingestion.validator import validate_video_path
from src.ingestion.video_loader import load_video


def test_valid_video():
    video = load_video("data/input/test.mp4")

    assert video.path == Path("data/input/test.mp4")
    assert video.duration > 0
    assert video.fps > 0
    assert video.width > 0
    assert video.height > 0
    assert video.video_codec


def test_missing_video():
    with pytest.raises(FileNotFoundError):
        validate_video_path("data/input/does_not_exist.mp4")


def test_unsupported_extension(tmp_path):
    fake_file = tmp_path / "test.txt"
    fake_file.write_text("not a video")

    with pytest.raises(ValueError):
        validate_video_path(fake_file)