import json

import pytest

from src.schemas.transcript import TranscriptResult, TranscriptSegment
from src.transcription.transcriber import transcribe_audio
from src.transcription.writer import save_transcript_json


def test_missing_audio():
    with pytest.raises(FileNotFoundError):
        transcribe_audio("data/input/does_not_exist.wav")


def test_save_transcript_json(tmp_path):
    transcript = TranscriptResult(
        language="en",
        language_probability=0.95,
        segments=[
            TranscriptSegment(
                start=0.0,
                end=5.0,
                text="Hello world.",
            ),
            TranscriptSegment(
                start=5.0,
                end=10.0,
                text="This is a test.",
            ),
        ],
    )

    output_path = tmp_path / "transcript.json"

    result = save_transcript_json(
        transcript=transcript,
        output_path=output_path,
    )

    assert result.exists()
    assert result.suffix == ".json"


def test_transcript_json_content(tmp_path):
    transcript = TranscriptResult(
        language="en",
        language_probability=0.90,
        segments=[
            TranscriptSegment(
                start=1.5,
                end=4.2,
                text="Testing timestamps.",
            )
        ],
    )

    output_path = tmp_path / "transcript.json"

    save_transcript_json(
        transcript=transcript,
        output_path=output_path,
    )

    with output_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["language"] == "en"
    assert data["language_probability"] == 0.90

    assert len(data["segments"]) == 1

    assert data["segments"][0]["start"] == 1.5
    assert data["segments"][0]["end"] == 4.2
    assert data["segments"][0]["text"] == "Testing timestamps."