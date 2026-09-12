from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.schemas.section import TranscriptSection
from src.schemas.transcript import TranscriptSegment


TRANSITION_PHRASES = (
    "let us discuss",
    "let's discuss",
    "now let's",
    "next",
    "moving on",
    "what was",
    "what is",
    "how is",
    "how's",
    "finally",
)


CONTINUATION_WORDS = (
    "and ",
    "or ",
    "but ",
    "while ",
    "including ",
    "which ",
    "that ",
    "because ",
    "therefore ",
    "it ",
    "this ",
    "these ",
    "they ",
    "its ",
    "their ",
)


def _build_context(
    segments: list[TranscriptSegment],
    start: int,
    end: int,
) -> str:
    return " ".join(
        segment.text
        for segment in segments[start:end]
    )


def _calculate_boundary_similarities(
    segments: list[TranscriptSegment],
    context_size: int = 3,
) -> list[float]:
    if len(segments) < 2:
        return []

    context_pairs = []

    for boundary in range(1, len(segments)):
        left_start = max(0, boundary - context_size)
        right_end = min(len(segments), boundary + context_size)

        left_text = _build_context(
            segments,
            left_start,
            boundary,
        )

        right_text = _build_context(
            segments,
            boundary,
            right_end,
        )

        context_pairs.append((left_text, right_text))

    all_texts = []

    for left_text, right_text in context_pairs:
        all_texts.extend([left_text, right_text])

    vectorizer = TfidfVectorizer(
        stop_words="english",
    )

    vectors = vectorizer.fit_transform(all_texts)

    similarities = []

    for index in range(0, len(all_texts), 2):
        similarity = cosine_similarity(
            vectors[index],
            vectors[index + 1],
        )[0][0]

        similarities.append(float(similarity))

    return similarities


def _is_likely_continuation(text: str) -> bool:
    text = text.strip()

    if not text:
        return True

    lower_text = text.lower()

    if lower_text.startswith(CONTINUATION_WORDS):
        return True

    if text[0].islower():
        return True

    return False


def _has_transition_phrase(text: str) -> bool:
    text = text.strip().lower()

    return text.startswith(TRANSITION_PHRASES)


def _create_section(
    segments: list[TranscriptSegment],
) -> TranscriptSection:
    text = " ".join(
        segment.text
        for segment in segments
    )

    return TranscriptSection(
        start=segments[0].start,
        end=segments[-1].end,
        text=text,
        source_segments=segments,
    )


def segment_transcript(
    segments: list[TranscriptSegment],
    similarity_threshold: float = 0.10,
    min_section_duration: float = 30.0,
    max_section_duration: float = 120.0,
    context_size: int = 3,
) -> list[TranscriptSection]:

    if not segments:
        return []

    similarities = _calculate_boundary_similarities(
        segments=segments,
        context_size=context_size,
    )

    sections = []
    section_start_index = 0

    for boundary_index, similarity in enumerate(
        similarities,
        start=1,
    ):
        current_start = segments[section_start_index].start
        current_end = segments[boundary_index - 1].end

        current_duration = current_end - current_start

        low_similarity = (
            similarity < similarity_threshold
        )

        long_enough = (
            current_duration >= min_section_duration
        )

        too_long = (
            current_duration >= max_section_duration
        )

        next_segment_text = segments[boundary_index].text

        continuation = _is_likely_continuation(
            next_segment_text
        )

        transition = _has_transition_phrase(
            next_segment_text
        )

        should_split = (
            not continuation
            and (
                (low_similarity and long_enough)
                or (transition and long_enough)
                or too_long
            )
        )

        if should_split:
            section_segments = segments[
                section_start_index:boundary_index
            ]

            sections.append(
                _create_section(section_segments)
            )

            section_start_index = boundary_index

    remaining_segments = segments[section_start_index:]

    if remaining_segments:
        sections.append(
            _create_section(remaining_segments)
        )

    return sections