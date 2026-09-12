
from pathlib import Path

import pytest

from src.preprocessing.audio import extract_audio
from src.preprocessing.frames import extract_frame, sample_frames


TEST_VIDEO = Path("data/input/test.mp4")


def test_extract_audio(tmp_path):
    output_path = tmp_path / "audio.wav"

    result = extract_audio(
        video_path=TEST_VIDEO,
        output_path=output_path,
    )

    assert result.exists()
    assert result.suffix == ".wav"
    assert result.stat().st_size > 0


def test_extract_frame(tmp_path):
    output_path = tmp_path / "frame.jpg"

    result = extract_frame(
        video_path=TEST_VIDEO,
        timestamp=5,
        output_path=output_path,
    )

    assert result.exists()
    assert result.suffix == ".jpg"
    assert result.stat().st_size > 0


def test_sample_frames(tmp_path):
    frames = sample_frames(
        video_path=TEST_VIDEO,
        duration=25,
        interval=10,
        output_dir=tmp_path,
    )

    assert len(frames) == 3

    assert frames[0].timestamp == 0
    assert frames[1].timestamp == 10
    assert frames[2].timestamp == 20

    for frame in frames:
        assert frame.path.exists()


def test_invalid_sampling_interval(tmp_path):
    with pytest.raises(ValueError):
        sample_frames(
            video_path=TEST_VIDEO,
            duration=20,
            interval=0,
            output_dir=tmp_path,
        )


def test_sample_frames_avoids_container_duration_tail(monkeypatch, tmp_path):
    extracted = []

    def fake_extract_frame(video_path, timestamp, output_path):
        extracted.append(timestamp)
        return output_path

    monkeypatch.setattr(
        "src.preprocessing.frames.extract_frame",
        fake_extract_frame,
    )

    frames = sample_frames(
        video_path="video.mp4",
        duration=20.01,
        interval=10.0,
        output_dir=tmp_path,
    )

    assert extracted == [0.0, 10.0]
    assert [frame.timestamp for frame in frames] == [0.0, 10.0]
