from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import traceback
import uuid

from main import run_pipeline
from src.providers.config import ExecutionProfile, ProviderConfig
from src.providers.hardware import detect_hardware


@dataclass
class ProcessingResult:
    run_id: str
    status: str
    output_dir: Path
    pdf_path: Path | None = None
    sections_count: int | None = None
    visual_candidates_count: int | None = None
    knowledge_visuals_count: int | None = None
    provider_summary: dict[str, str] | None = None
    elapsed_seconds: float | None = None
    error: str | None = None
    technical_details: str | None = None


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "video"
    suffix = Path(name).suffix.lower()
    return f"{stem[:80]}{suffix}"


def create_run_directory(output_base: str | Path) -> tuple[str, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
    output_base = Path(output_base).resolve()
    run_dir = (output_base / run_id).resolve()
    if output_base not in run_dir.parents:
        raise ValueError("Run directory escaped the configured output directory.")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def save_uploaded_file(run_dir: Path, filename: str, content: bytes) -> Path:
    safe_name = sanitize_filename(filename)
    input_dir = (run_dir / "input").resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    target = (input_dir / safe_name).resolve()
    if input_dir not in target.parents:
        raise ValueError("Unsafe upload filename.")
    target.write_bytes(content)
    return target


def _arguments(
    input_path: Path,
    output_dir: Path,
    profile: ExecutionProfile,
    provider_config: ProviderConfig | None,
    options: dict,
) -> Namespace:
    config = provider_config
    hardware = detect_hardware()
    default_device = "cuda" if hardware.cuda_available else "cpu"
    default_compute = "float16" if hardware.cuda_available else "int8"
    return Namespace(
        input=str(input_path), output=str(output_dir), language=options.get("language", "en"),
        model_size=options.get("model_size", config.asr_model if config else "medium"),
        device=options.get("device", config.device if config else default_device),
        compute_type=options.get("compute_type", config.compute_type if config else default_compute),
        frame_interval=options.get("frame_interval", 10.0),
        scene_threshold=options.get("scene_threshold", 0.03),
        scene_sample_interval=options.get("scene_sample_interval", 1.0),
        min_candidate_gap=options.get("min_candidate_gap", 2.0),
        max_visuals=options.get("max_visuals", 40), start=options.get("start", 0.0),
        duration=options.get("duration"), mode=(ExecutionProfile.CUSTOM.value if config else profile.value),
        vision_provider=config.vision_provider.value if config else None,
        synthesis_provider=config.synthesis_provider.value if config else None,
        verification_provider=config.verification_provider.value if config else None,
        vision_model=config.vision_model if config else None,
        synthesis_model=config.synthesis_model if config else None,
        verification_model=config.verification_model if config else None,
        provider_config=config,
    )


def run_video2knowledge(
    input_source: str | Path,
    output_dir: str | Path,
    *,
    execution_profile: ExecutionProfile = ExecutionProfile.SMART,
    provider_config: ProviderConfig | None = None,
    resume: bool = True,
    progress_callback=None,
    **options,
) -> ProcessingResult:
    input_path = Path(input_source).resolve()
    run_dir = Path(output_dir).resolve()
    run_id = run_dir.name
    if not input_path.is_file():
        return ProcessingResult(run_id, "FAILED", run_dir, error="Input video was not found.")
    if execution_profile is ExecutionProfile.CUSTOM and provider_config is None:
        return ProcessingResult(run_id, "FAILED", run_dir, error="Advanced mode requires provider selections.")
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        data = run_pipeline(
            _arguments(input_path, run_dir, execution_profile, provider_config, options),
            progress_callback=progress_callback,
        )
        return ProcessingResult(
            run_id=run_id, status=data["evaluation"]["status"], output_dir=run_dir,
            pdf_path=Path(data["pdf_path"]).resolve(), sections_count=data["sections_count"],
            visual_candidates_count=data["visual_candidates_count"],
            knowledge_visuals_count=data["knowledge_visuals_count"],
            provider_summary=data["provider_summary"], elapsed_seconds=data["elapsed_seconds"],
        )
    except Exception as error:
        message = str(error)
        normalized = message.lower()
        if "out of memory" in normalized or "cuda_error_out_of_memory" in normalized:
            message = "The selected local model does not fit available GPU memory. Choose a smaller model or CPU mode."
        elif "cuda" in normalized and ("not available" in normalized or "driver" in normalized):
            message = "CUDA is unavailable. Select CPU mode or check the NVIDIA driver/runtime."
        return ProcessingResult(
            run_id, "FAILED", run_dir, error=message,
            technical_details=traceback.format_exc(),
        )
