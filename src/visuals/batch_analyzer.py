from src.schemas.transcript import TranscriptSegment
from src.schemas.visual import RankedVisual, VisualAnalysis
from src.visuals.context import get_transcript_context
from src.visuals.vlm_analyzer import analyze_visual
from src.visuals.writer import load_visual_analyses, upsert_visual_analysis


def analyze_ranked_visuals(
    ranked_visuals: list[RankedVisual],
    transcript_segments: list[TranscriptSegment],
    context_window: float = 10.0,
    checkpoint_path=None,
    provider=None,
) -> list[VisualAnalysis]:

    results = (
        load_visual_analyses(checkpoint_path)
        if checkpoint_path is not None
        else []
    )
    completed = {round(item.timestamp, 3) for item in results}

    for index, visual in enumerate(
        ranked_visuals,
        start=1,
    ):
        if round(visual.timestamp, 3) in completed:
            continue
        print(
            f"Analyzing {index}/{len(ranked_visuals)} "
            f"at {visual.timestamp:.1f}s..."
        )

        transcript_context = get_transcript_context(
            transcript_segments,
            timestamp=visual.timestamp,
            window=context_window,
        )

        result = (
            provider.analyze(visual.path, visual.timestamp, transcript_context)
            if provider is not None
            else analyze_visual(
                image_path=visual.path,
                timestamp=visual.timestamp,
                transcript_context=transcript_context,
            )
        )

        results.append(result)
        if checkpoint_path is not None:
            upsert_visual_analysis(result, checkpoint_path)

    return results
