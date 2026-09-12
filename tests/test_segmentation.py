from src.schemas.transcript import TranscriptSegment
from src.segmentation.segmenter import segment_transcript


def test_empty_transcript():
    sections = segment_transcript([])

    assert sections == []


def test_single_topic_segments():
    segments = [
        TranscriptSegment(
            start=0.0,
            end=10.0,
            text="Heart attack prediction uses medical claims.",
        ),
        TranscriptSegment(
            start=10.0,
            end=20.0,
            text="Medical claims contain diagnosis information.",
        ),
        TranscriptSegment(
            start=20.0,
            end=30.0,
            text="Diagnosis information can indicate patient risk.",
        ),
    ]

    sections = segment_transcript(
        segments,
        min_section_duration=30.0,
    )

    assert len(sections) == 1

    assert sections[0].start == 0.0
    assert sections[0].end == 30.0
    assert len(sections[0].source_segments) == 3


def test_transition_phrase_creates_boundary():
    segments = [
        TranscriptSegment(
            start=0.0,
            end=10.0,
            text="The dataset contains medical claims.",
        ),
        TranscriptSegment(
            start=10.0,
            end=20.0,
            text="Patients have diagnosis and pharmacy records.",
        ),
        TranscriptSegment(
            start=20.0,
            end=30.0,
            text="The records cover multiple years.",
        ),
        TranscriptSegment(
            start=30.0,
            end=40.0,
            text="Let us discuss how we aggregate this data.",
        ),
        TranscriptSegment(
            start=40.0,
            end=50.0,
            text="The codes are grouped into categories.",
        ),
    ]

    sections = segment_transcript(
        segments,
        similarity_threshold=0.0,
        min_section_duration=30.0,
        max_section_duration=120.0,
    )

    assert len(sections) == 2

    assert sections[0].start == 0.0
    assert sections[0].end == 30.0

    assert sections[1].start == 30.0
    assert sections[1].end == 50.0


def test_continuation_does_not_create_bad_boundary():
    segments = [
        TranscriptSegment(
            start=0.0,
            end=10.0,
            text="The target variable represents heart attack occurrence.",
        ),
        TranscriptSegment(
            start=10.0,
            end=20.0,
            text="The target is binary.",
        ),
        TranscriptSegment(
            start=20.0,
            end=30.0,
            text="It is represented using plus one or minus one.",
        ),
        TranscriptSegment(
            start=30.0,
            end=40.0,
            text="This indicates whether the event occurred.",
        ),
    ]

    sections = segment_transcript(
        segments,
        similarity_threshold=1.0,
        min_section_duration=20.0,
    )

    assert sections[0].start == 0.0

    assert not any(
        section.start == 20.0
        for section in sections
    )

    assert not any(
        section.start == 30.0
        for section in sections
    )


def test_section_preserves_source_segments():
    original_segments = [
        TranscriptSegment(
            start=0.0,
            end=5.0,
            text="First sentence.",
        ),
        TranscriptSegment(
            start=5.0,
            end=10.0,
            text="Second sentence.",
        ),
    ]

    sections = segment_transcript(
        original_segments,
        min_section_duration=30.0,
    )

    assert len(sections) == 1

    section = sections[0]

    assert section.source_segments == original_segments
    assert "First sentence." in section.text
    assert "Second sentence." in section.text