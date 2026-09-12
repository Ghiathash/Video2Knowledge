from pathlib import Path

from faster_whisper import WhisperModel

from src.schemas.transcript import TranscriptResult, TranscriptSegment


def transcribe_audio(
    audio_path: str | Path,
    model_size: str = "medium",
    device: str = "cuda",
    compute_type: str = "float16",
    language: str | None = None,
) -> TranscriptResult:

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )

    segments_generator, info = model.transcribe(
        str(audio_path),
        language=language,
        task="transcribe",
        beam_size=5,
        vad_filter=True,
    )

    segments = []

    for segment in segments_generator:
        segments.append(
            TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
            )
        )

    return TranscriptResult(
        language=info.language,
        language_probability=info.language_probability,
        segments=segments,
    )