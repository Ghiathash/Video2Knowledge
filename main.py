import argparse
import json
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

from src.ingestion.video_loader import load_video
from src.preprocessing.media_preprocessor import preprocess_video

from src.transcription.transcriber import transcribe_audio
from src.transcription.writer import save_transcript_json

from src.segmentation.segmenter import segment_transcript

from src.visuals.scene_detector import detect_candidate_frames
from src.visuals.transcript_guided import (
    extract_transcript_guided_candidates,
)
from src.visuals.candidate_fusion import merge_candidates
from src.visuals.visual_deduplicator import deduplicate_candidates
from src.visuals.visual_ranker import rank_candidate_frames
from src.visuals.batch_analyzer import analyze_ranked_visuals
from src.visuals.writer import save_visual_analyses

from src.alignment.aligner import align_visuals_to_sections
from src.alignment.writer import save_aligned_sections

from src.synthesis.synthesizer import synthesize_report
from src.synthesis.writer import save_knowledge_report

from src.verification.faithfulness import verify_report
from src.verification.writer import save_verification_audit

from src.reporting.pdf_generator import generate_pdf_report

from src.evaluation.report_evaluator import evaluate_report
from src.schemas.section import TranscriptSection
from src.schemas.transcript import TranscriptResult, TranscriptSegment
from src.schemas.visual import CandidateFrame, RankedVisual
from src.visuals.writer import load_visual_analyses


def parse_args():
    parser = argparse.ArgumentParser(
        description="Video2Knowledge full pipeline"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input video path",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Independent output directory for this run",
    )

    parser.add_argument(
        "--language",
        default="en",
        help="ASR language. Use 'auto' for automatic detection.",
    )

    parser.add_argument(
        "--model-size",
        default="medium",
        help="faster-whisper model size",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
    )

    parser.add_argument(
        "--compute-type",
        default="float16",
    )

    parser.add_argument(
        "--frame-interval",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=0.03,
        help="Visual scene-change threshold",
    )

    parser.add_argument(
        "--scene-sample-interval",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--min-candidate-gap",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--max-visuals",
        type=int,
        default=40,
        help=(
            "Maximum number of ranked candidates sent "
            "to the VLM. Use 0 for unlimited."
        ),
    )

    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Optional start position in seconds",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help=(
            "Optional duration in seconds. "
            "Useful for testing part of a very long video."
        ),
    )

    return parser.parse_args()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return _jsonable(
            asdict(value)
        )

    if hasattr(value, "model_dump"):
        return _jsonable(
            value.model_dump()
        )

    if isinstance(value, dict):
        return {
            str(k): _jsonable(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item)
            for item in value
        ]

    return value


def _save_json(
    data,
    output_path: Path,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            _jsonable(data),
            file,
            ensure_ascii=False,
            indent=2,
        )


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _valid_json(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        _load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return True


def _load_transcript(path: Path) -> TranscriptResult:
    data = _load_json(path)
    return TranscriptResult(
        language=data["language"],
        language_probability=float(data["language_probability"]),
        segments=[TranscriptSegment(**item) for item in data["segments"]],
    )


def _load_sections(path: Path) -> list[TranscriptSection]:
    return [
        TranscriptSection(
            start=float(item["start"]),
            end=float(item["end"]),
            text=item["text"],
            source_segments=[
                TranscriptSegment(**segment)
                for segment in item.get("source_segments", [])
            ],
        )
        for item in _load_json(path)
    ]


def _load_candidates(path: Path) -> list[CandidateFrame]:
    return [
        CandidateFrame(
            path=Path(item["path"]),
            timestamp=float(item["timestamp"]),
            change_score=float(item.get("change_score", 0.0)),
            sources=item.get("sources", []),
            evidence=item.get("evidence"),
        )
        for item in _load_json(path)
    ]


def _load_ranked(path: Path) -> list[RankedVisual]:
    return [
        RankedVisual(
            path=Path(item["path"]),
            timestamp=float(item["timestamp"]),
            change_score=float(item["change_score"]),
            visual_richness_score=float(item["visual_richness_score"]),
            importance_score=float(item["importance_score"]),
        )
        for item in _load_json(path)
    ]


def _stage(
    number: int,
    total: int,
    name: str,
):
    print()
    print("=" * 65)
    print(
        f"[{number}/{total}] {name}"
    )
    print("=" * 65)


def _make_clip_if_needed(
    input_path: Path,
    output_root: Path,
    start: float,
    duration: float | None,
) -> Path:

    if (
        start <= 0
        and duration is None
    ):
        return input_path

    clip_dir = (
        output_root
        / "_input"
    )

    clip_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    clip_path = (
        clip_dir
        / "working_clip.mp4"
    )

    if clip_path.is_file() and clip_path.stat().st_size > 0:
        print(f"Reusing working clip: {clip_path}")
        return clip_path

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    if start > 0:
        command.extend(
            [
                "-ss",
                str(start),
            ]
        )

    command.extend(
        [
            "-i",
            str(input_path),
        ]
    )

    if duration is not None:
        command.extend(
            [
                "-t",
                str(duration),
            ]
        )

    command.extend(
        [
            "-c",
            "copy",
            str(clip_path),
        ]
    )

    print(
        "Preparing test clip..."
    )

    subprocess.run(
        command,
        check=True,
    )

    return clip_path


def _find_audio(
    output_root: Path,
) -> Path:

    preferred = (
        output_root
        / "audio"
        / "audio.wav"
    )

    if preferred.exists():
        return preferred

    audio_dir = (
        output_root
        / "audio"
    )

    candidates = list(
        audio_dir.glob("*.wav")
    )

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        "Could not locate extracted WAV audio "
        f"inside: {audio_dir}"
    )


def main():
    args = parse_args()

    total_stages = 11

    input_path = Path(
        args.input
    )

    output_root = Path(
        args.output
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input video not found: {input_path}"
        )

    run_start = time.perf_counter()

    working_video = (
        _make_clip_if_needed(
            input_path=input_path,
            output_root=output_root,
            start=args.start,
            duration=args.duration,
        )
    )

    # ========================================================
    # 1. Ingestion
    # ========================================================

    _stage(
        1,
        total_stages,
        "Video Ingestion",
    )

    video = load_video(
        working_video
    )

    print(
        f"Path: {video.path}"
    )
    print(
        f"Duration: {video.duration:.2f}s"
    )
    print(
        f"FPS: {video.fps:.2f}"
    )
    print(
        f"Resolution: "
        f"{video.width}x{video.height}"
    )

    # ========================================================
    # 2. Preprocessing
    # ========================================================

    _stage(
        2,
        total_stages,
        "Media Preprocessing",
    )

    if not (output_root / "audio" / "audio.wav").is_file():
        preprocess_video(
            video_path=working_video,
            output_dir=output_root,
            frame_interval=args.frame_interval,
        )
    else:
        print("Reusing preprocessing outputs.")

    audio_path = _find_audio(
        output_root
    )

    print(
        f"Audio: {audio_path}"
    )

    # ========================================================
    # 3. ASR
    # ========================================================

    _stage(
        3,
        total_stages,
        "Speech-to-Text",
    )

    language = (
        None
        if args.language.lower()
        == "auto"
        else args.language
    )

    transcript_path = (
        output_root
        / "transcript"
        / "transcript.json"
    )
    if _valid_json(transcript_path):
        transcript = _load_transcript(transcript_path)
        print("Reusing transcript checkpoint.")
    else:
        transcript = transcribe_audio(
            audio_path=audio_path,
            model_size=args.model_size,
            device=args.device,
            compute_type=args.compute_type,
            language=language,
        )
        save_transcript_json(transcript, transcript_path)

    print(
        f"Segments: "
        f"{len(transcript.segments)}"
    )

    # ========================================================
    # 4. Segmentation
    # ========================================================

    _stage(
        4,
        total_stages,
        "Transcript Segmentation",
    )

    sections_path = (
        output_root
        / "segmentation"
        / "sections.json"
    )
    if _valid_json(sections_path):
        sections = _load_sections(sections_path)
        print("Reusing segmentation checkpoint.")
    else:
        sections = segment_transcript(
            transcript.segments,
            similarity_threshold=0.10,
            min_section_duration=30.0,
            max_section_duration=120.0,
            context_size=3,
        )
        _save_json(sections, sections_path)

    print(
        f"Sections: {len(sections)}"
    )

    # ========================================================
    # 5. Hybrid Candidate Generation
    # ========================================================

    _stage(
        5,
        total_stages,
        "Hybrid Visual Candidate Generation",
    )

    candidate_root = (
        output_root
        / "candidates_hybrid"
    )

    candidates_path = output_root / "visuals" / "candidates.json"
    if _valid_json(candidates_path):
        unique_candidates = _load_candidates(candidates_path)
        scene_candidates = []
        transcript_candidates = []
        merged_candidates = unique_candidates
        print("Reusing candidate checkpoint.")
    else:
        scene_candidates = detect_candidate_frames(
            video_path=working_video,
            output_dir=(
                candidate_root
                / "scene"
            ),
            sample_interval=(
                args.scene_sample_interval
            ),
            change_threshold=(
                args.scene_threshold
            ),
            min_candidate_gap=(
                args.min_candidate_gap
            ),
        )

        transcript_candidates = extract_transcript_guided_candidates(
            video_path=working_video,
            transcript_segments=(
                transcript.segments
            ),
            output_dir=(
                candidate_root
                / "transcript"
            ),
        )

        merged_candidates = merge_candidates(
            scene_candidates=(
                scene_candidates
            ),
            transcript_candidates=(
                transcript_candidates
            ),
            video_duration=(
                video.duration
            ),
            tolerance=2.0,
            start_margin=1.0,
            end_margin=2.0,
        )

        unique_candidates = deduplicate_candidates(
            merged_candidates,
            similarity_threshold=0.98,
        )
        _save_json(unique_candidates, candidates_path)

    print(
        f"Scene candidates: "
        f"{len(scene_candidates)}"
    )
    print(
        f"Transcript candidates: "
        f"{len(transcript_candidates)}"
    )
    print(
        f"Merged: "
        f"{len(merged_candidates)}"
    )
    print(
        f"Unique: "
        f"{len(unique_candidates)}"
    )

    # ========================================================
    # 6. Ranking
    # ========================================================

    _stage(
        6,
        total_stages,
        "Visual Importance Ranking",
    )

    top_k = (
        None
        if args.max_visuals <= 0
        else args.max_visuals
    )

    ranked_path = output_root / "visuals" / "ranked_visuals.json"
    if _valid_json(ranked_path):
        ranked_visuals = _load_ranked(ranked_path)
        print("Reusing ranking checkpoint.")
    else:
        ranked_visuals = rank_candidate_frames(
            candidates=(
                unique_candidates
            ),
            video_duration=(
                video.duration
            ),
            end_margin=2.0,
            top_k=top_k,
        )
        _save_json(ranked_visuals, ranked_path)

    print(
        f"Ranked visuals: "
        f"{len(ranked_visuals)}"
    )

    # ========================================================
    # 7. VLM Understanding
    # ========================================================

    _stage(
        7,
        total_stages,
        "Visual Understanding",
    )

    visual_analysis_path = output_root / "visuals" / "visual_analysis.json"
    if _valid_json(visual_analysis_path):
        knowledge_visuals = load_visual_analyses(visual_analysis_path)
        analyses = knowledge_visuals
        print("Reusing visual-analysis checkpoint.")
    else:
        analyses = analyze_ranked_visuals(
            ranked_visuals=(
                ranked_visuals
            ),
            transcript_segments=(
                transcript.segments
            ),
            context_window=10.0,
            checkpoint_path=visual_analysis_path,
        )
        knowledge_visuals = [visual for visual in analyses if visual.contains_knowledge]
        save_visual_analyses(knowledge_visuals, visual_analysis_path)

    print(
        f"Analyzed: "
        f"{len(analyses)}"
    )
    print(
        f"Knowledge visuals: "
        f"{len(knowledge_visuals)}"
    )

    # ========================================================
    # 8. Alignment
    # ========================================================

    _stage(
        8,
        total_stages,
        "Transcript-Visual Alignment",
    )

    aligned_path = output_root / "alignment" / "aligned_sections.json"
    if _valid_json(aligned_path):
        aligned_sections = _load_json(aligned_path)
        print("Reusing alignment checkpoint.")
    else:
        aligned_sections = align_visuals_to_sections(
            sections=sections,
            visuals=(
                knowledge_visuals
            ),
            boundary_window=5.0,
            semantic_weight=0.8,
        )
        save_aligned_sections(aligned_sections, aligned_path)

    print(
        f"Aligned sections: "
        f"{len(aligned_sections)}"
    )

    # ========================================================
    # 9. Synthesis + Verification
    # ========================================================

    _stage(
        9,
        total_stages,
        "Knowledge Synthesis + Faithfulness Verification",
    )

    report_dir = (
        output_root
        / "report"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    draft_report_path = (
        report_dir
        / "knowledge_report.json"
    )

    if _valid_json(draft_report_path):
        print("Reusing synthesis checkpoint.")
    else:
        draft_report = synthesize_report(
            str(aligned_path),
            checkpoint_path=str(report_dir / "synthesis_checkpoint.json"),
        )
        save_knowledge_report(draft_report, draft_report_path)

    verified_report_path = report_dir / "verified_report.json"
    audit_path = report_dir / "verification_audit.json"
    if _valid_json(verified_report_path) and _valid_json(audit_path):
        print("Reusing verified report checkpoint.")
    else:
        verified_report, audit = verify_report(
            aligned_sections_path=(
                str(aligned_path)
            ),
            draft_report_path=(
                str(draft_report_path)
            ),
            checkpoint_path=str(
                report_dir
                / "verification_checkpoint.json"
            ),
        )
        save_knowledge_report(verified_report, verified_report_path)
        save_verification_audit(audit, audit_path)

    print(
        "Verified report created."
    )

    # ========================================================
    # 10. PDF
    # ========================================================

    _stage(
        10,
        total_stages,
        "PDF Generation",
    )

    pdf_path = (
        report_dir
        / "video2knowledge_report.pdf"
    )

    generate_pdf_report(
        verified_report_path=(
            str(verified_report_path)
        ),
        aligned_sections_path=(
            str(aligned_path)
        ),
        output_path=(
            str(pdf_path)
        ),
        source_video_path=(
            str(working_video)
        ),
    )

    print(
        f"PDF: {pdf_path}"
    )

    # ========================================================
    # 11. Evaluation
    # ========================================================

    _stage(
        11,
        total_stages,
        "Evaluation / QA",
    )

    evaluation_path = (
        report_dir
        / "evaluation_report.json"
    )

    evaluation = evaluate_report(
        verified_report_path=(
            str(verified_report_path)
        ),
        aligned_sections_path=(
            str(aligned_path)
        ),
        verification_audit_path=(
            str(audit_path)
        ),
        pdf_path=str(
            pdf_path
        ),
        output_path=str(
            evaluation_path
        ),
    )

    elapsed = (
        time.perf_counter()
        - run_start
    )

    print()
    print("=" * 65)
    print("VIDEO2KNOWLEDGE COMPLETE")
    print("=" * 65)

    print(
        f"Status: "
        f"{evaluation['status']}"
    )

    print(
        f"Sections: "
        f"{evaluation['metrics']['sections']}"
    )

    print(
        f"Visuals: "
        f"{evaluation['metrics']['report_visuals']}"
    )

    print(
        f"Elapsed: "
        f"{elapsed / 60:.1f} minutes"
    )

    print(
        f"Output: {output_root}"
    )

    print(
        f"PDF: {pdf_path}"
    )


if __name__ == "__main__":
    main()
